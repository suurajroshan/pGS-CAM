import h5py

def read_txt_file(file_path, num_points: np.int32):
  data_app = []
  labels_app = []
  for f in file_path:
      with open(f) as file:
          count=0
          data = []
          labels = []
          for line in file:
              values = line.strip().split(' ')
              data.append([np.float32(value) for value in values[:-1]])
              labels.append(np.int32(float(values[-1])))
          data = np.array(data)
          labels = np.array(labels)
          data_app = data if len(data_app) == 0 else np.dstack((data_app, data))
          labels_app = labels if len(labels_app) == 0 else np.dstack((labels_app, labels))
  return np.transpose(data_app, (2,0,1)), np.squeeze(np.transpose(labels_app))


def getDataFiles(data_path, num_points:int):
  output_file = 'out.h5'

  str_fname = [line.rstrip() for line in open(os.path.join(data_path, 'synsetoffset2category.txt'))][0]
  sub_filename = str_fname.split()[1]

  json_files = glob.glob(os.path.join(data_path, 'train_test_split', '*.json'))
  if os.path.isfile(os.path.join(data_path, output_file)) is False:
    for file in json_files:
      fname = file.split('/')[-1]
      stage_name = fname.split('_')[1]
      f = open(file)
      jsonf = json.load(f)
      file_arr = [line.rstrip().split('/')[-1] for line in jsonf]
      file_paths = [os.path.join(data_path, sub_filename,i+'.txt') for i in file_arr]
      out_data, out_labels = read_txt_file(file_path=file_paths, num_points=num_points)
      
      with h5py.File(os.path.join(data_path, output_file), 'a') as hfile:
        group = hfile.create_group(str(stage_name))
        group.create_dataset('data', data=out_data)
        group.create_dataset('labels', data=out_labels)
        print('%s data group created'%str(stage_name))

def load_h5(data_path, stage_name):
  file = os.path.join(data_path, 'out.h5')
  f = h5py.File(file, 'r')
  data = f[str(stage_name)]['data'][:]
  label = f[str(stage_name)]['labels'][:]
  return (data, label)

def loadDataFile(data_path, stage_name):
  return load_h5(data_path, stage_name)


def load_h5_data_label_seg(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:] # (2048, 2048, 3)
  seg = f['labels'][:] # (2048, 2048)
  return (data, seg)