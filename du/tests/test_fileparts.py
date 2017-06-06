from unittest import TestCase
import du 

class TestFileparts(TestCase):
  def test_fileparts_is_accurate(self):
    f1 = '/a/b/c.txt'
    f2 = '../d.jpeg'

    b1, n1, e1 = du.fileparts(f1)
    b2, n2, e2 = du.fileparts(f2)

    self.assertTrue(b1 == '/a/b')
    self.assertTrue(n1 == 'c')
    self.assertTrue(e1 == '.txt')

    self.assertTrue(b2 == '..')
    self.assertTrue(n2 == 'd')
    self.assertTrue(e2 == '.jpeg')
