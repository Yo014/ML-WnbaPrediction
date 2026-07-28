import sys
import os
import train_totals_model

class FlushingStream:
    def __init__(self, target, filepath):
        self.target = target
        self.file = open(filepath, 'w')
    def write(self, s):
        self.target.write(s)
        self.target.flush()
        self.file.write(s)
        self.file.flush()
        os.fsync(self.file.fileno())
    def flush(self):
        self.target.flush()
        self.file.flush()

sys.stdout = FlushingStream(sys.stdout, 'console.log')
sys.stderr = FlushingStream(sys.stderr, 'console.log')

print("=== STARTING TRAIN TOTALS MODEL ===")
try:
    train_totals_model.train_totals_model()
    print("=== FINISHED TRAIN TOTALS MODEL ===")
except Exception as e:
    import traceback
    print("=== ERROR IN TRAIN TOTALS MODEL ===")
    print(e)
    traceback.print_exc()
