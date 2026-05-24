from unittest.mock import MagicMock, patch

import trainlib.plugins as plugins
from trainlib.recipes import recipe_registry


def _reset_discovered():
    plugins._discovered = False


class TestDiscoverPlugins:
    def setup_method(self):
        _reset_discovered()

    def teardown_method(self):
        _reset_discovered()

    @patch("trainlib.plugins.entry_points")
    def test_loads_recipe_entry_point(self, mock_eps):
        mock_fn = MagicMock()
        ep = MagicMock()
        ep.name = "test-plugin-recipe"
        ep.load.return_value = mock_fn

        mock_eps.return_value = {ep}

        def side_effect(group):
            if group == "trainlib.recipes":
                return [ep]
            return []

        mock_eps.side_effect = side_effect

        plugins.discover_plugins()

        assert recipe_registry.has("test-plugin-recipe")
        assert recipe_registry.get("test-plugin-recipe") is mock_fn

        # Cleanup
        recipe_registry._items.pop("test-plugin-recipe", None)

    @patch("trainlib.plugins.entry_points")
    def test_skips_broken_entry_point(self, mock_eps):
        broken_ep = MagicMock()
        broken_ep.name = "broken-plugin"
        broken_ep.load.side_effect = ImportError("no such module")

        def side_effect(group):
            if group == "trainlib.recipes":
                return [broken_ep]
            return []

        mock_eps.side_effect = side_effect

        plugins.discover_plugins()
        assert not recipe_registry.has("broken-plugin")

    @patch("trainlib.plugins.entry_points")
    def test_idempotent(self, mock_eps):
        mock_eps.return_value = []
        mock_eps.side_effect = lambda group: []

        plugins.discover_plugins()
        plugins.discover_plugins()

        # entry_points is called for each group only on first call
        assert mock_eps.call_count == 4  # 4 groups, called once

    @patch("trainlib.plugins.entry_points")
    def test_no_entry_points_is_noop(self, mock_eps):
        mock_eps.side_effect = lambda group: []

        plugins.discover_plugins()
        # No error, no registrations
