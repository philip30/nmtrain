import chainer
import importlib
import numpy
import unittest
import sys

import nmtrain.environment as environment

class TestInitGpu(unittest.TestCase):
  def setUp(self):
    importlib.reload(chainer)
    importlib.reload(environment)

  @unittest.skipIf(not hasattr(chainer.cuda, "cupy"), "No GPU detected")
  def test_gpu0(self):
    environment.init_gpu(0)
    self.assertEqual(environment.array_module(), chainer.cuda.cupy)

  def test_cpu(self):
    environment.init_gpu(-1)
    self.assertEqual(environment.array_module(), numpy)

  def test_nogpu(self):
    cupy = None
    if hasattr(chainer.cuda, "cupy"):
      cupy = getattr(chainer.cuda, "cupy")
      delattr(chainer.cuda, "cupy")
    environment.init_gpu(0)
    self.assertEqual(environment.array_module(), numpy)
    if cupy is not None:
      setattr(chainer.cuda, "cupy", cupy)

  def test_premature_call(self):
    temp = sys.stderr
    sys.stderr = None
    try:
      module = environment.array_module()
      self.assertFalse(True)
    except:
      self.assertTrue(True)
    sys.stderr = temp

class TestRandom(unittest.TestCase):
  def setUp(self):
    importlib.reload(environment)
    importlib.reload(numpy)
    importlib.reload(chainer)
    # Do not change this seed
    self.seed = 11

  def test_random_cpu(self):
    environment.init_gpu(-1)
    environment.init_random(self.seed)
    self.assertAlmostEqual(float(environment.array_module().random.random()),
                           0.1802696888767692)

  @unittest.skipIf(not hasattr(chainer.cuda, "cupy"), "No GPU detected.")
  def test_random_gpu(self):
    environment.init_gpu(0)
    environment.init_random(self.seed)
    self.assertAlmostEqual(float(environment.array_module().random.random()),
                           0.5405254640354904)
    # TODO(philip30):
    # Seems that the cuda.cupy.random draws from a different distribution than
    # the one in numpy. For now it is important to first init all model using
    # One of the module and convert to the other to ensure that it is producing
    # the same result every time training is conducted

if __name__ == '__main__':
    unittest.main()
