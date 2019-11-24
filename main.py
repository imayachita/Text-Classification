from models import TextCatModel
import time
from datetime import timedelta

start_time = time.time()

text_cat = TextCatModel(config_file='config.json')
text_cat.training()
text_cat.evaluate()

print ('Elapsed Time:', str(timedelta(seconds=(time.time()-start_time))))
