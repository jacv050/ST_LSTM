import os
import sys
import multiprocessing as mp

def process_sk_fn(fn, save_dir):
    fid = open(fn, 'r')
    wfid = open(os.path.join(save_dir, os.path.basename(fn)),'w')
    header = fid.readline()
    cnt = int(header.strip())
    fn_joints = []
    has_nan = False
    for i in range(cnt):
        header = fid.readline()
        bd_cnt = int(header.strip())
        if has_nan:
            break

        for j in range(bd_cnt):
            ignore = fid.readline()
            joints = int(fid.readline().strip())
            joints_2d = []
            for k in range(joints):
                values = fid.readline().strip().split()
                if values[5] == 'NaN' or values[6] == 'NaN':
                    has_nan = True
                joints_2d.append(values[5])
                joints_2d.append(values[6])
            if j == 0:
                # only keep the first one in the model.
                #fn_joints.append(joints_2d)
                wfid.write(' '.join(joints_2d) + '\n')

    fid.close()
    wfid.close()
    if has_nan:
        os.remove(os.path.join(save_dir, os.path.basename(fn)))


def worker(lst_fns, save_dir, thread_id):
    for idx, fn in enumerate(lst_fns):
        process_sk_fn(fn, save_dir)
        if (idx % 200 == 0):
            print('thread {:d}, idx {:d}'.format(thread_id, idx))
            sys.stdout.flush()



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {0} <src_dir> <dst_dir> [num_threads = 10] [missing_fn=./Matlab/samples_with_missing_skeletons.txt]'.format(sys.argv[0]))
        sys.exit()
    num_threads = 10
    missing_fn = './Matlab/samples_with_missing_skeletons.txt'
    if(len(sys.argv) >= 4):
        num_threads = int(sys.argv[3])
    if(len(sys.argv) >= 5):
        missing_fn= sys.argv[4]

    dict_missing_key = {}
    with open(missing_fn, 'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            dict_missing_key[parts[0]] = 1

    src_dir = sys.argv[1]
    save_dir = sys.argv[2]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dict_fns = {}

    idx = 0
    for root, subdirs, fns in os.walk(src_dir):
        for fn in fns:
            key = os.path.splitext(fn)[0]
            if key in dict_missing_key:
               continue

            if (idx % num_threads) not in dict_fns:
                dict_fns[idx % num_threads] = []
            dict_fns[idx % num_threads].append(os.path.join(root, fn))
            idx += 1

    processes = [mp.Process(target = worker, args = (dict_fns[i], save_dir, i + 1)) for i in range(num_threads)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

