from dask.distributed import LocalCluster, Client
from dask.distributed import as_completed
import time
import dask.array as da
import imageio
import os
import numpy as np


def readImg(path):
    img = imageio.imread(path)
    im = np.array(img,dtype='uint8')
    return im

def writeImg(path,buf):
    imageio.imwrite(path,buf)

def part_median_filter(buf, x):
    nx=buf.shape[0]
    ny=buf.shape[1]
    
    #
    # CREATE NEW LINE WITH MEDIAN FILTER SOLUTION
    #
    new_buf=np.zeros([ny,3],dtype='uint8')
    
    ##########################################
    #
    # TODO COMPUTE MEDIAN FILTER
    #
    neighbor = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
    neighbor_top = [[0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
    neighbor_bot = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1]]

    for y in range(1,ny-1):
        r = 0
        g = 0
        b = 0
        if x ==0:
            for x_n, y_n in neighbor_top:
                r = r + buf[x_n,y+y_n,0]
                g = g + buf[x_n,y+y_n,1]
                b = b + buf[x_n,y+y_n,2]
            new_buf[y,0] = r/len(neighbor_top)
            new_buf[y,1] = g/len(neighbor_top)
            new_buf[y,2] = b/len(neighbor_top)

        elif x == int(nx)-1:
            for x_n, y_n in neighbor_bot:
                r = r + buf[ x+x_n,y+y_n,0]
                g = g + buf[ x+x_n,y+y_n,1]
                b = b + buf[ x+x_n,y+y_n,2]
            new_buf[y,0] = r/len(neighbor_bot)
            new_buf[y,1] = g/len(neighbor_bot)
            new_buf[y,2] = b/len(neighbor_bot)

        else:
            for x_n, y_n in neighbor:
                r = r + buf[ x+x_n,y+y_n,0]
                g = g + buf[ x+x_n,y+y_n,1]
                b = b + buf[ x+x_n,y+y_n,2]
            new_buf[y,0] = r/len(neighbor)
            new_buf[y,1] = g/len(neighbor)
            new_buf[y,2] = b/len(neighbor)
    

    ##########################################
    #
    # RETURN LOCAL IMAGE PART
    #
    return [x, new_buf]

def main():
    data_dir = './data'
    file = os.path.join(data_dir,'lena_noisy.jpg')
    img_buf=readImg(file)
    img_buf=np.array(img_buf)
    print('SHAPE',img_buf.shape)
    nx=img_buf.shape[0]
    ny=img_buf.shape[1]
    
    
    ###########################################################################
    #
    # CREATE DASK Client
    cluster = LocalCluster()
    client = Client(cluster)
    c = Client(n_workers=4)

    ###########################################################################
    #
    # PARALLEL MEDIAN FILTER COMPUTATION
    
    futures = [c.submit(part_median_filter, img_buf, x) for x in range(nx)]
    iterator = as_completed(futures)
    
    new_buf=np.zeros([nx,ny,3],dtype='uint8')
    
    ###########################################################################
    #
    # COMPUTE NEW IMAGE RESULTS FROM RESULT RDD
    for res in iterator:
        result = res.result()
        new_buf[result[0],:,:] = result[1]
    
    print('CREATE NEW PICTURE FILE')
    filter_file = os.path.join(data_dir,'dask_lena_filter.jpg')
    writeImg(filter_file,new_buf)

if __name__ == '__main__':
    main()
