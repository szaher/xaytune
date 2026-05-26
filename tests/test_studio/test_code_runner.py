from __future__ import annotations

from xaytune.studio.code_runner import CODE_TEMPLATES, CodeResult, run_code


class TestCodeResult:
    def test_defaults(self):
        r = CodeResult()
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.error is None
        assert r.duration == 0.0


class TestRunCode:
    def test_simple_print(self):
        result = run_code('print("hello")')
        assert result.stdout.strip() == "hello"
        assert result.error is None

    def test_captures_stdout(self):
        result = run_code("for i in range(3): print(i)")
        assert "0" in result.stdout
        assert "1" in result.stdout
        assert "2" in result.stdout

    def test_captures_stderr(self):
        result = run_code('import sys; sys.stderr.write("warn\\n")')
        assert "warn" in result.stderr

    def test_syntax_error(self):
        result = run_code("def foo(")
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self):
        result = run_code("1 / 0")
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_name_error(self):
        result = run_code("print(undefined_var)")
        assert result.error is not None
        assert "NameError" in result.error

    def test_duration_tracked(self):
        result = run_code("x = sum(range(1000))")
        assert result.duration >= 0.0

    def test_xaytune_in_namespace(self):
        result = run_code("print(xaytune.__version__)")
        assert result.error is None
        assert result.stdout.strip() != ""

    def test_custom_namespace(self):
        result = run_code("print(my_var)", namespace={"my_var": 42})
        assert result.stdout.strip() == "42"
        assert result.error is None

    def test_timeout(self):
        result = run_code(
            "import time; time.sleep(10)",
            timeout=0.1,
        )
        assert result.error is not None
        assert "timed out" in result.error.lower()

    def test_empty_code(self):
        result = run_code("")
        assert result.error is None
        assert result.stdout == ""

    def test_multiline(self):
        code = "x = 1\ny = 2\nprint(x + y)"
        result = run_code(code)
        assert result.stdout.strip() == "3"


class TestCodeTemplates:
    def test_all_templates_are_strings(self):
        for name, tmpl in CODE_TEMPLATES.items():
            assert isinstance(tmpl, str), f"Template '{name}' is not a string"
            assert len(tmpl) > 0

    def test_expected_templates_exist(self):
        assert "Fine-tuning" in CODE_TEMPLATES
        assert "LoRA Fine-tuning" in CODE_TEMPLATES
        assert "Alignment (DPO)" in CODE_TEMPLATES
        assert "Alignment (GRPO)" in CODE_TEMPLATES
        assert "Pre-training" in CODE_TEMPLATES
        assert "Custom" in CODE_TEMPLATES

    def test_templates_contain_import(self):
        for name, tmpl in CODE_TEMPLATES.items():
            assert "import xaytune" in tmpl, f"Template '{name}' missing import"

    def test_templates_are_valid_syntax(self):
        for name, tmpl in CODE_TEMPLATES.items():
            try:
                compile(tmpl, f"<template:{name}>", "exec")
            except SyntaxError:
                raise AssertionError(f"Template '{name}' has invalid syntax")
