import pyspark
from pyspark import SparkContext
import imageio
import os
import numpy as np

def readImg(path):
    img = imageio.imread(path)
    im = np.array(img,dtype='uint8')
    return im

def writeImg(path,buf):
    imageio.imwrite(path,buf)

def part_median_filter(local_data):
    part_id = local_data[0]
    first   = local_data[1]
    end     = local_data[2]
    buf     = local_data[3]
    nx=buf.shape[0]
    ny=buf.shape[1]
    
    ########################################
    #
    # CREATE NEW BUF WITH MEDIAN FILTER SOLUTION
    #
    new_buf=np.zeros([int(end-first),ny,3],dtype='uint8')
    
    ##########################################
    #
    # TODO COMPUTE MEDIAN FILTER
    #
    neighbor = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
    neighbor_top = [[0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
    neighbor_bot = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1]]
    for x in range(int(end-first)):
        for y in range(1,ny-1):
            r = 0
            g = 0
            b = 0
            if int(first) == 0 and x ==0: #for the first row
                for x_n, y_n in neighbor_top:
                    r = r + buf[x_n,y+y_n,0]
                    g = g + buf[x_n,y+y_n,1]
                    b = b + buf[x_n,y+y_n,2]
                new_buf[0,y,0] = r/len(neighbor_top)
                new_buf[0,y,1] = g/len(neighbor_top)
                new_buf[0,y,2] = b/len(neighbor_top)
                
            elif int(end) == int(ny) and x == int(end-first)-1: #for the last row
                for x_n, y_n in neighbor_bot:
                    r = r + buf[int(first) + x+x_n,y+y_n,0]
                    g = g + buf[int(first) + x+x_n,y+y_n,1]
                    b = b + buf[int(first) + x+x_n,y+y_n,2]
                new_buf[x,y,0] = r/len(neighbor_bot)
                new_buf[x,y,1] = g/len(neighbor_bot)
                new_buf[x,y,2] = b/len(neighbor_bot)
                
            else: # for the other row
                for x_n, y_n in neighbor:
                    r = r + buf[int(first) + x+x_n,y+y_n,0]
                    g = g + buf[int(first) + x+x_n,y+y_n,1]
                    b = b + buf[int(first) + x+x_n,y+y_n,2]
                new_buf[x,y,0] = r/len(neighbor)
                new_buf[x,y,1] = g/len(neighbor)
                new_buf[x,y,2] = b/len(neighbor)
    
    ##########################################
    #
    # RETURN LOCAL IMAGE PART
    #
    return part_id,new_buf

def main():
    data_dir = 'data'
    file = os.path.join(data_dir,'lena_noisy.jpg')
    img_buf=readImg(file)
    print('SHAPE',img_buf.shape)
    print('IMG\n',img_buf)
    nx=img_buf.shape[0]
    ny=img_buf.shape[1]
    
    ###########################################################################
    #
    # SPLT IMAGES IN NB_PARTITIONS PARTS
    nb_partitions = 8
    print("NB PARTITIONS : ",nb_partitions)
    data=[]
    begin=0
    block_size=nx/nb_partitions
    for ip in range(nb_partitions):
        end=min(begin+block_size,nx)
        data.append([ip,begin,end,img_buf])
        begin=end   
    
    ###########################################################################
    #
    # CREATE SPARKCONTEXT
    sc =SparkContext()
    data_rdd = sc.parallelize(data,nb_partitions)	
    
    ###########################################################################
    #
    # PARALLEL MEDIAN FILTER COMPUTATION
    result_rdd = data_rdd.map(part_median_filter)
    result_data = result_rdd.collect()

    new_img_buf=np.zeros([nx,ny,3],dtype='uint8')
    ###########################################################################
    #
    # COMPUTE NEW IMAGE RESULTS FROM RESULT RDD
    # TODO
    for i in range(len(result_data)):
        #print(result_data[i][0])
        new_img_buf[int(block_size*i):int(block_size*(i+1)),:,:] = result_data[i][1]
    
    print('CREATE NEW PICTURE FILE')
    filter_file = os.path.join(data_dir,'spark_lena_filter.jpg')
    writeImg(filter_file,new_img_buf)

if __name__ == '__main__':
    main()