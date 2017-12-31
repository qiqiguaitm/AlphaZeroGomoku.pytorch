import os
import shutil
import uuid
import cPickle as pickle
import time

TunnelPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
if not os.path.exists(TunnelPath):
    os.mkdir(TunnelPath)


def touch(fname):
    open(fname, 'a').close()
    os.utime(fname, None)


def upload(data_server_url, file_path):
    cmd_upload = 'curl -F "file=@%s" %s' % (file_path, data_server_url)
    os.system(cmd_upload)


def download(data_server_url, file_name, save_path):
    if '/' in file_name:
        file_name = os.path.split(file_name)[-1]
    cmd_download = 'wget %s/%s -O %s --timeout=600 ' % (data_server_url, file_name, save_path + '.tmp')
    os.system(cmd_download)
    if os.path.exists(save_path + '.tmp'):
        size = os.path.getsize(save_path + '.tmp')
    else:
        return False
    if size == 0:
        os.remove(save_path + '.tmp')
        return False
    else:
        shutil.move(save_path + '.tmp', save_path)
        return True


def upload_samples(data_server_url, samples):
    tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp/')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    samples_path = os.path.join(tmp_dir, 'samples.' + str(int(time.time())) + '_' + str(uuid.uuid4()) + '.pkl')
    with open(samples_path, 'wb') as pickle_file:
        pickle.dump(samples, pickle_file, 2)
    upload(data_server_url, samples_path)
    os.remove(samples_path)


def download_samples():
    samples = []
    fs = os.listdir(TunnelPath)
    for f in fs:
        if not f.startswith('samples.'):
            continue
        try:
            with open(os.path.join(TunnelPath, f), 'r') as pickle_file:
                ss = pickle.load(pickle_file)
                samples.extend(ss)
            os.remove(os.path.join(TunnelPath, f))
            return samples
        except Exception, e:
            print e
            continue
    return samples


if __name__ == '__main__':
    DIST_DATA_URL = 'http://10.83.150.55:8000/'
    while True:
        s = [1, 2, 3]
        upload_samples(DIST_DATA_URL, s)
        time.sleep(2)

    samples = download_samples()
    print samples
