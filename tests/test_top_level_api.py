import trainlib
from trainlib import export


class TestTopLevelAPI:
    def test_version(self):
        assert trainlib.__version__ == "0.1.0"

    def test_finetune_importable(self):
        assert callable(trainlib.finetune)

    def test_pretrain_importable(self):
        assert callable(trainlib.pretrain)

    def test_align_importable(self):
        assert callable(trainlib.align)

    def test_evaluate_importable(self):
        assert callable(trainlib.evaluate)

    def test_export_module(self):
        assert callable(export.merge)
        assert callable(export.save)
        assert callable(export.push_to_hub)

    def test_finetune_is_recipe(self):
        from trainlib.recipes.finetune import finetune

        assert trainlib.finetune is finetune

    def test_pretrain_is_recipe(self):
        from trainlib.recipes.pretrain import pretrain

        assert trainlib.pretrain is pretrain

    def test_align_is_recipe(self):
        from trainlib.recipes.align.align import align

        assert trainlib.align is align

    def test_evaluate_is_eval(self):
        from trainlib.eval.evaluate import evaluate

        assert trainlib.evaluate is evaluate
