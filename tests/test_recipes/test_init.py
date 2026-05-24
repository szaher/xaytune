from xaytune.recipes import finetune, pretrain, recipe_registry


class TestRecipesPublicAPI:
    def test_finetune_importable(self):
        assert callable(finetune)

    def test_pretrain_importable(self):
        assert callable(pretrain)

    def test_recipe_registry_has_finetune(self):
        assert recipe_registry.has("finetune")
        assert recipe_registry.get("finetune") is finetune

    def test_recipe_registry_has_pretrain(self):
        assert recipe_registry.has("pretrain")
        assert recipe_registry.get("pretrain") is pretrain

    def test_recipe_registry_list(self):
        recipes = recipe_registry.list()
        assert "finetune" in recipes
        assert "pretrain" in recipes
