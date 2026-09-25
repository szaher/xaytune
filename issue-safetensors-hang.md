## Native worker can deadlock while loading a model (safetensors 0.8.0 / transformers 5.17.0)

### What happened

During a full local test run on `main`-plus-PR-013b (no host/runtime/worker code changed), one test timed out:

```
FAILED tests/test_experiment/test_restart_reconciliation.py::test_a_settled_experiment_attaches_without_its_runtime[runtime-unavailable]
asyncio.exceptions.TimeoutError   # handle.wait() under asyncio.wait_for(..., timeout=180)
1 failed, 2551 passed, 4 skipped, 10 deselected
```

The test trains the ~4k-parameter tiny GPT-2 fixture for 2 steps, which normally takes seconds. Rerun alone it passed 3/3 (both parameters).

The worker had not crashed. It was **hung at 0% CPU**, and the test's timeout left it **orphaned**: the launcher (`python -m xaytune.runtimes.local.launcher …`, PPID 1) and its worker (`xaytune.workers.native main`) were still alive four and a half minutes later, after pytest had already deleted the workload directory. Nothing reaps a worker whose controller gave up.

### Where it is stuck

`sample` on the worker (macOS), two threads of interest:

- **Main thread**: in a Python-level lock wait (`acquire_timed`), then `PyEval_RestoreThread → take_gil`, waiting for the GIL.
- **A loader thread** (transformers' threaded weight loading): inside safetensors' Rust extension:

```
safetensors_rust::PySafeSlice::__getitem__
  std::sync::once_lock::OnceLock<T>::initialize
    pyo3::sync::once_lock::init_once_cell_py_attached
      once_cell::imp::initialize_or_wait
        <pyo3::internal::state::SuspendAttach as Drop>::drop
          PyEval_RestoreThread → take_gil → _pthread_cond_wait
```

Both are waiting for the GIL. The loader thread holds a `OnceLock` initialisation while it waits to re-attach; the pattern looks like the lock-ordering deadlock between a pyo3 once-cell initialised with the GIL released and another thread needing that cell. The full sample is attached.

### Versions

- safetensors 0.8.0
- transformers 5.17.0
- torch 2.14 (local), Python 3.10.15, macOS (Darwin 25.0.0, arm64)

### Why it matters beyond the test

The same load path runs in real training (native worker; likely TRL too). A worker that deadlocks during model load never reports anything, so the attempt stays running until something external gives up, and the process is then orphaned rather than cleaned up.

### Reproduce

Intermittent; seen once in a ~7-minute full run under load (builds running concurrently):

```
XAYTUNE_REQUIRE_TRL=1 .venv/bin/pytest tests/ -q -m "not slow"
```

Looping the single test may reproduce it faster:

```
for i in $(seq 50); do .venv/bin/pytest -q "tests/test_experiment/test_restart_reconciliation.py::test_a_settled_experiment_attaches_without_its_runtime" || break; done
```

### Possible directions (not decided)

- Check whether a newer safetensors release fixes the pyo3 once-cell deadlock, or pin one that predates it.
- Disable transformers' threaded loading in the workers, if 5.17 exposes a switch.
- Separately: a controller-side timeout or cancellation should not leave the launcher and worker orphaned.

<details><summary>Worker call graph (macOS <code>sample</code>, 2 s, all threads)</summary>

```
Call graph:
    1758 Thread_24199907: Main Thread
    + 1758 start  (in dyld) + 7184  [0x183e81d54]
    +   1758 Py_BytesMain  (in Python) + 40  [0x102cde3bc]
    +     1758 Py_RunMain  (in Python) + 340  [0x102cdd1b8]
    +       1758 PyRun_SimpleStringFlags  (in Python) + 64  [0x102cc596c]
    +         1758 PyRun_StringFlags  (in Python) + 128  [0x102cc5a34]
    +           1758 run_mod  (in Python) + 112  [0x102cc402c]
    +             1758 run_eval_code_obj  (in Python) + 84  [0x102cc40c8]
    +               1758 PyEval_EvalCode  (in Python) + 104  [0x102c76138]
    +                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                   1758 _PyEval_EvalFrameDefault  (in Python) + 31044  [0x102c7eae4]
    +                     1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                       1758 _PyEval_EvalFrameDefault  (in Python) + 30764  [0x102c7e9cc]
    +                         1758 call_function  (in Python) + 128  [0x102c8259c]
    +                           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                             1758 _PyEval_EvalFrameDefault  (in Python) + 30764  [0x102c7e9cc]
    +                               1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                   1758 _PyEval_EvalFrameDefault  (in Python) + 31044  [0x102c7eae4]
    +                                     1758 PyVectorcall_Call  (in Python) + 176  [0x102baa3c0]
    +                                       1758 method_vectorcall  (in Python) + 124  [0x102bacbb0]
    +                                         1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                           1758 _PyEval_EvalFrameDefault  (in Python) + 31044  [0x102c7eae4]
    +                                             1758 PyVectorcall_Call  (in Python) + 176  [0x102baa3c0]
    +                                               1758 method_vectorcall  (in Python) + 124  [0x102bacbb0]
    +                                                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                   1758 _PyEval_EvalFrameDefault  (in Python) + 30584  [0x102c7e918]
    +                                                     1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                       1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                         1758 _PyEval_EvalFrameDefault  (in Python) + 30764  [0x102c7e9cc]
    +                                                           1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                             1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                               1758 _PyEval_EvalFrameDefault  (in Python) + 30764  [0x102c7e9cc]
    +                                                                 1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                                   1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                                     1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                                                                       1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                                         1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                                           1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                                                                             1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                                               1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                                                 1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                                                                                   1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                                                     1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                                                       1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                                                                                         1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                                                           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                                                             1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                                                                                               1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                                                                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                                                                                   1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                                                                                                     1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                                                                                       1758 method_vectorcall_VARARGS_KEYWORDS  (in Python) + 152  [0x102bb58ec]
    +                                                                                                         1758 lock_PyThread_acquire_lock  (in Python) + 56  [0x102d1de18]
    +                                                                                                           1758 acquire_timed  (in Python) + 200  [0x102d1dc70]
    +                                                                                                             1758 PyEval_RestoreThread  (in Python) + 24  [0x102c75ba0]
    +                                                                                                               1724 take_gil  (in Python) + 520  [0x102c75498]
    +                                                                                                               ! 1693 _pthread_cond_wait  (in libsystem_pthread.dylib) + 984  [0x1842410dc]
    +                                                                                                               ! : 1693 __psynch_cvwait  (in libsystem_kernel.dylib) + 8  [0x1842014f8]
    +                                                                                                               ! 31 _pthread_cond_wait  (in libsystem_pthread.dylib) + 340  [0x184240e58]
    +                                                                                                               !   31 __gettimeofday  (in libsystem_kernel.dylib) + 12  [0x184201b4c]
    +                                                                                                               17 take_gil  (in Python) + 416  [0x102c75430]
    +                                                                                                               ! 17 gettimeofday  (in libsystem_c.dylib) + 56  [0x1840d3d60]
    +                                                                                                               !   11 __commpage_gettimeofday_internal  (in libsystem_kernel.dylib) + 44  [0x1841ffc10]
    +                                                                                                               !   : 11 mach_absolute_time  (in libsystem_kernel.dylib) + 108  [0x1841fe0fc]
    +                                                                                                               !   6 __commpage_gettimeofday_internal  (in libsystem_kernel.dylib) + 0  [0x1841ffbe4]
    +                                                                                                               17 take_gil  (in Python) + 384,520  [0x102c75410,0x102c75498]
    1758 Thread_24199968
    + 1758 thread_start  (in libsystem_pthread.dylib) + 8  [0x18423bba8]
    +   1758 _pthread_start  (in libsystem_pthread.dylib) + 136  [0x184240c08]
    +     1758 pythread_wrapper  (in Python) + 48  [0x102cd07e8]
    +       1758 thread_run  (in Python) + 120  [0x102d1e8e4]
    +         1758 method_vectorcall  (in Python) + 392  [0x102baccbc]
    +           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +             1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +               1758 call_function  (in Python) + 128  [0x102c8259c]
    +                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                   1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                     1758 call_function  (in Python) + 128  [0x102c8259c]
    +                       1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                         1758 _PyEval_EvalFrameDefault  (in Python) + 31044  [0x102c7eae4]
    +                           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                             1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                               1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                   1758 _PyEval_EvalFrameDefault  (in Python) + 31044  [0x102c7eae4]
    +                                     1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                       1758 _PyEval_EvalFrameDefault  (in Python) + 30636  [0x102c7e94c]
    +                                         1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                             1758 _PyEval_EvalFrameDefault  (in Python) + 7424  [0x102c78ea0]
    +                                               1758 pyo3::impl_::trampoline::binaryfunc::he6e1580e7d82a970  (in _safetensors_rust.abi3.so) + 72  [0x112c4a1b4]
    +                                                 1758 safetensors_rust::PySafeSlice::__pymethod___getitem____::hbce87da1dba35a92  (in _safetensors_rust.abi3.so) + 176  [0x112c35f0c]
    +                                                   1758 safetensors_rust::PySafeSlice::__getitem__::he36f59aaf4431294  (in _safetensors_rust.abi3.so) + 2072  [0x112c32760]
    +                                                     1758 std::sync::once_lock::OnceLock$LT$T$GT$::initialize::h5cc7656c8a65bf93  (in _safetensors_rust.abi3.so) + 40  [0x112caf154]
    +                                                       1758 _RNvMNtNtNtNtCsg55jX0GwzBC_3std3sys4sync4once5queueNtB2_4Once4call  (in _safetensors_rust.abi3.so) + 256  [0x112cb8df8]
    +                                                         1758 std::sync::once::Once::call_once_force::_$u7b$$u7b$closure$u7d$$u7d$::h1d1041f7e3a8ede6  (in _safetensors_rust.abi3.so) + 604  [0x112c4deb0]
    +                                                           1758 pyo3::sync::once_lock::init_once_cell_py_attached::h2f096db7a0838c71  (in _safetensors_rust.abi3.so) + 104  [0x112caf454]
    +                                                             1758 once_cell::imp::OnceCell$LT$T$GT$::initialize::h6b3b228c0a00c0b9  (in _safetensors_rust.abi3.so) + 72  [0x112cafc10]
    +                                                               1758 once_cell::imp::initialize_or_wait::h867407d446aa8791  (in _safetensors_rust.abi3.so) + 324  [0x112c7928c]
    +                                                                 1758 once_cell::imp::OnceCell$LT$T$GT$::initialize::_$u7b$$u7b$closure$u7d$$u7d$::h4979409cc3601a5c  (in _safetensors_rust.abi3.so) + 52  [0x112c52f08]
    +                                                                   1758 _$LT$pyo3..internal..state..SuspendAttach$u20$as$u20$core..ops..drop..Drop$GT$::drop::h12efd45fe493bfb9  (in _safetensors_rust.abi3.so) + 44  [0x112c76888]
    +                                                                     1758 PyEval_RestoreThread  (in Python) + 24  [0x102c75ba0]
    +                                                                       1758 take_gil  (in Python) + 520  [0x102c75498]
    +                                                                         1730 _pthread_cond_wait  (in libsystem_pthread.dylib) + 984  [0x1842410dc]
    +                                                                         ! 1730 __psynch_cvwait  (in libsystem_kernel.dylib) + 8  [0x1842014f8]
    +                                                                         17 _pthread_cond_wait  (in libsystem_pthread.dylib) + 340  [0x184240e58]
    +                                                                         ! 17 __gettimeofday  (in libsystem_kernel.dylib) + 12  [0x184201b4c]
    +                                                                         6 _pthread_cond_wait  (in libsystem_pthread.dylib) + 1124  [0x184241168]
    +                                                                         ! 6 _pthread_mutex_firstfit_lock_slow  (in libsystem_pthread.dylib) + 220  [0x18423b868]
    +                                                                         !   6 _pthread_mutex_firstfit_lock_wait  (in libsystem_pthread.dylib) + 84  [0x18423de3c]
    +                                                                         !     6 __psynch_mutexwait  (in libsystem_kernel.dylib) + 8  [0x1842009c8]
    +                                                                         5 _pthread_cond_wait  (in libsystem_pthread.dylib) + 1092  [0x184241148]
    +                                                                           5 _pthread_cond_updateval  (in libsystem_pthread.dylib) + 0  [0x18423dbd8]
    1758 Thread_24199969
    + 1758 thread_start  (in libsystem_pthread.dylib) + 8  [0x18423bba8]
    +   1758 _pthread_start  (in libsystem_pthread.dylib) + 136  [0x184240c08]
    +     1758 pythread_wrapper  (in Python) + 48  [0x102cd07e8]
    +       1758 thread_run  (in Python) + 120  [0x102d1e8e4]
    +         1758 method_vectorcall  (in Python) + 392  [0x102baccbc]
    +           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +             1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +               1758 call_function  (in Python) + 128  [0x102c8259c]
    +                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                   1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                     1758 call_function  (in Python) + 128  [0x102c8259c]
    +                       1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                         1758 _PyEval_EvalFrameDefault  (in Python) + 31044  [0x102c7eae4]
    +                           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                             1758 _PyEval_EvalFrameDefault  (in Python) + 30468  [0x102c7e8a4]
    +                               1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                 1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                   1758 _PyEval_EvalFrameDefault  (in Python) + 31044  [0x102c7eae4]
    +                                     1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                       1758 _PyEval_EvalFrameDefault  (in Python) + 30636  [0x102c7e94c]
    +                                         1758 call_function  (in Python) + 128  [0x102c8259c]
    +                                           1758 _PyEval_Vector  (in Python) + 396  [0x102c762d8]
    +                                             1758 _PyEval_EvalFrameDefault  (in Python) + 7424  [0x102c78ea0]
    +                                               1758 pyo3::impl_::trampoline::binaryfunc::he6e1580e7d82a970  (in _safetensors_rust.abi3.so) + 72  [0x112c4a1b4]
    +                                                 1758 safetensors_rust::PySafeSlice::__pymethod___getitem____::hbce87da1dba35a92  (in _safetensors_rust.abi3.so) + 176  [0x112c35f0c]
    +                                                   1758 safetensors_rust::PySafeSlice::__getitem__::he36f59aaf4431294  (in _safetensors_rust.abi3.so) + 2072  [0x112c32760]
    +                                                     1758 std::sync::once_lock::OnceLock$LT$T$GT$::initialize::h5cc7656c8a65bf93  (in _safetensors_rust.abi3.so) + 40  [0x112caf154]
    +                                                       1758 _RNvMNtNtNtNtCsg55jX0GwzBC_3std3sys4sync4once5queueNtB2_4Once4call  (in _safetensors_rust.abi3.so) + 200  [0x112cb8dc0]
    +                                                         1758 _RNvNtNtNtNtCsg55jX0GwzBC_3std3sys4sync4once5queue4wait  (in _safetensors_rust.abi3.so) + 360  [0x112c97db0]
    +                                                           1758 _dispatch_semaphore_wait_slow  (in libdispatch.dylib) + 132  [0x184086f40]
    +                                                             1758 _dispatch_sema4_wait  (in libdispatch.dylib) + 28  [0x184086990]
    +                                                               1758 semaphore_wait_trap  (in libsystem_kernel.dylib) + 8  [0x1841fdbb0]
    1758 Thread_24200568
      1758 start_wqthread  (in libsystem_pthread.dylib) + 8  [0x18423bb9c]
        1758 _pthread_wqthread  (in libsystem_pthread.dylib) + 368  [0x18423ce98]
          1758 __workq_kernreturn  (in libsystem_kernel.dylib) + 8  [0x1841ff9dc]
```

</details>
