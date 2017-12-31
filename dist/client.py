import os
import shutil
import uuid
import cPickle as pickle

TunnelPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/')
if not os.path.exists(TunnelPath):
    os.mkdir(TunnelPath)

def touch(fname):
    open(fname, 'a').close()
    os.utime(fname, None)


data_server_url = 'http://10.83.150.55:8000/'

def upload(data_server_url,file_path):
    cmd_upload = 'curl -F "file=@%s" %s'%(file_path,data_server_url)
    os.system(cmd_upload)

def download(data_server_url,file_name,save_path):
    cmd_download = 'wget %s/%s -O %s --timeout=600 '%(data_server_url,file_name,save_path)
    os.system(cmd_download)

def upload_samples(data_server_url,samples):
    tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp/')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    samples_path = os.path.join(tmp_dir,'samples.'+str(uuid.uuid4())+'.pkl')
    with open(samples_path, 'wb') as pickle_file:
        pickle.dump(samples,pickle_file,2)
    upload(data_server_url,samples_path)
    os.remove(samples_path)

def download_samples():
    samples = []
    fs = os.listdir(TunnelPath)
    for f in fs:
        if not f.startswith('samples.'):
            continue
        try:
            with open(os.path.join(TunnelPath,f),'r') as pickle_file:
                ss = pickle.load(pickle_file)
                samples.extend(ss)
            os.remove(os.path.join(TunnelPath,f))
        except Exception,e:
            print e
            continue
    return samples

if __name__ == '__main__':
    s = [1,2,3]
    upload_samples(data_server_url,s)


    samples = download_samples()
    print samples





