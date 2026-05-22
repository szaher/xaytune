import trainlib


class TestTopLevelAPI:
    def test_version(self):
        assert trainlib.__version__ == "0.1.0"

    def test_finetune_importable(self):
        assert callable(trainlib.finetune)

    def test_pretrain_importable(self):
        assert callable(trainlib.pretrain)

    def test_finetune_is_recipe(self):
        from trainlib.recipes.finetune import finetune
        assert trainlib.finetune is finetune

    def test_pretrain_is_recipe(self):
        from trainlib.recipes.pretrain import pretrain
        assert trainlib.pretrain is pretrain
