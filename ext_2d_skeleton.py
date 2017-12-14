import os
import sys
import multiprocessing as mp

def process_sk_fn(fn, save_dir):
    fid = open(fn, 'r')
    wfid = open(save_dir, os.path.basename(fn),'w')
    header = fid.readline()
    cnt = int(header.strip())
    fn_joints = []
    for i in range(cnt):
        header = fid.readline()
        bd_cnt = int(header.strip())

        for j in range(bd_cnt):
            ignore = fid.readline()
            joints = int(fid.readline().strip())
            joints_2d = []
            for k in range(joints):
                values = fid.readline().strip().split()
                joints_2d.append(values[5])
                joints_2d.append(values[6])
            if j == 0:
                # only keep the first one in the model.
                #fn_joints.append(joints_2d)
                wfid.write(' '.join(joints_2d) + '\n')

    fid.close()
    wfid.close()


def worker(lst_fns, save_dir, thread_id):
    for idx, fn in enumerate(lst_fns):
        process_sk_fn(fn, save_dir)
        if (idx % 200 == 0):
            print('thread {d}, idx {d}', thread_id, idx)
            sys.stdout.flush()



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {0} <src_dir> <dst_dir> [num_threads = 10]'.format(sys.argv[0]))
        sys.exit()
    num_threads = 10
    if(len(sys.argv) == 4):
        num_threads = int(sys.argv[3])

    src_dir = sys.argv[1]
    save_dir = sys.argv[2]
    dict_fns = {}

    idx = 0
    for root, subdirs, fns in os.walk(src_dir):
        for fn in fns:
            if (idx % num_threads) not in dict_fns:
                dict_fns[idx % num_threads] = []
            dict_fns[idx % num_threads].append(os.path.join(root, fn))

    processes = [mp.Process(target = worker, args = (dict_fns[i], save_dir, i + 1)) for i in range(num_threads)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

