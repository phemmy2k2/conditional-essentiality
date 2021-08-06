# get params from form controls
t = 'heal/media/indir/'
#
import pandas as pd
def file_header(file):
    path = ''.join((t,file.name))
    f = pd.read_csv(path)
    # f = pd.DataFrame(file.name)
    cols = f.columns
    cols = cols.to_list()
    return cols

# if __name__ == '__main__':
#     file_header(file)