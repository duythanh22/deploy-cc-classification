import time
import litserve as ls

class PredictTimeLogger(ls.Callback):
    def on_before_predict(self, lit_api):
        self.start_time = time.perf_counter()

    def on_after_predict(self, lit_api):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        print(f"Predict took {elapsed_time: .3f} seconds", flush=True)