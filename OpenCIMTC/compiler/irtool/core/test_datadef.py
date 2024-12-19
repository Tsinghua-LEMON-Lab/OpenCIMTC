from unittest import TestCase, main
from .datadef import DataDef


class TestDataDef(TestCase):

    def test_datadef(self):
        d = DataDef()
        self.assertIsNone(d.ref)
        self.assertIsNone(d.batch)
        self.assertIsNone(d.channel)
        self.assertIsNone(d.dims)
        self.assertIsNone(d.dtype)
        self.assertIsNone(d.channel_last)

        d = DataDef(channel_last=True)
        self.assertIs(d.channel_last, True)

        d = DataDef('foo')
        self.assertEqual(d.ref, 'foo')
        self.assertEqual(d.to_json_obj(), 'foo')


if __name__ == '__main__':
    main()
