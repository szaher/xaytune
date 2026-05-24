import pytest

from xaytune.utils.registry import Registry


class TestRegistry:
    def setup_method(self):
        self.registry = Registry("test")

    def test_register_and_get(self):
        @self.registry.register("my-item")
        def my_func():
            return 42

        assert self.registry.get("my-item") is my_func

    def test_register_returns_original(self):
        @self.registry.register("item")
        def my_func():
            return 1

        assert my_func() == 1

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found in test registry"):
            self.registry.get("nonexistent")

    def test_list_registered(self):
        @self.registry.register("a")
        def func_a():
            pass

        @self.registry.register("b")
        def func_b():
            pass

        assert self.registry.list() == ["a", "b"]

    def test_duplicate_raises(self):
        @self.registry.register("dup")
        def first():
            pass

        with pytest.raises(ValueError, match="already registered in test"):

            @self.registry.register("dup")
            def second():
                pass

    def test_register_class(self):
        @self.registry.register("my-class")
        class MyClass:
            pass

        assert self.registry.get("my-class") is MyClass

    def test_has(self):
        @self.registry.register("exists")
        def func():
            pass

        assert self.registry.has("exists") is True
        assert self.registry.has("missing") is False

    def test_register_with_override(self):
        @self.registry.register("item")
        def first():
            return 1

        @self.registry.register("item", override=True)
        def second():
            return 2

        assert self.registry.get("item") is second
