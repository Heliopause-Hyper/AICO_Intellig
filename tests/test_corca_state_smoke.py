import os
import sys
import unittest


class TestCORCAStateSmoke(unittest.TestCase):
    def test_run_default_template(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        exec_state_path = os.path.join(repo_root, "folderA", "preciseFZ", "apply", "exec_state")
        hdf5_file_path = os.path.join(repo_root, "folderA", "preciseFZ", "databank", "COMRES_last", "define_rod_01_001.hdf5")
        if not os.path.exists(exec_state_path):
            self.skipTest(f"exec_state not found: {exec_state_path}")
        if not os.path.exists(hdf5_file_path):
            self.skipTest(f"hdf5 not found: {hdf5_file_path}")

        sys.path.append(os.path.join(repo_root, "src"))
        from corcasim_simulator import run_corcasim_simulation

        result = run_corcasim_simulation(
            template_name="default",
            params={"Prk:": 50, "Pp:": 15.5, "Tin:": 292.1, "bore_ppm": 1000},
            exec_state_path=exec_state_path,
            hdf5_file_path=hdf5_file_path,
        )

        self.assertTrue(result.get("success"), msg=str(result))
        self.assertIn("keff", result)
        self.assertIn("FQ", result)
        self.assertIsInstance(result["keff"], float)
        self.assertIsInstance(result["FQ"], float)


if __name__ == "__main__":
    unittest.main()

