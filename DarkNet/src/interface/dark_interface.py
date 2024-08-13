from interface import DarkHelp
import json

class Net:
    def __init__(self, *net):
        print("DarkHelp Version:" + DarkHelp.DarkHelpVersion().decode("utf-8"))
        print("Darknet Version:" + DarkHelp.DarknetVersion().decode("utf-8"))
        self.dh = DarkHelp.CreateDarkHelpNN(*net)
        if not self.dh:
            print("""
                Failed to allocate a DarkHelp object.  Possible problems include:

                1) missing neural network files, or files are in a different directory.
                2) libraries needed by DarkHelp or Darknet have not been installed.
                3) errors in DarkHelp or Darknet libraries.""")
            quit(1)
    
    def predict(self, image):
        DarkHelp.Predict(self.dh, *image.shape[:2][::-1], image.tobytes(order='C'), image.size)
        return json.loads(DarkHelp.GetPredictionResults(self.dh).decode())
    
    def __del__(self):
        DarkHelp.DestroyDarkHelpNN(self.dh)