import tensorflow as tf
import numpy as np
#
# with tf.Session() as sess:
#     new_saver=tf.train.import_meta_graph('models/re.ckpt.meta')
#     new_saver.restore(sess,'models/re.ckpt')
#     graph=tf.get_default_graph()
#     y=tf.get_collection('pred_network')[0]
#     X=graph.get_operation_by_name('X').outputs[0]
#     input=np.ones([1,1,100,3])
#     print(sess.run(y,feed_dict={X:input}))


# import random
# def foo(n):
#     random.seed()
#     c1=0
#     c2=0
#     for i in range(n):
#         x=random.random()
#         y=random.random()
#         r1=x*x+y*y
#         r2=(1-x)*(1-x)+(1-y)*(1-y)
#         if r1<=1 and r2<=1:
#             c1+=1
#         else:
#             c2+=1
#     return c1/c2
#
# foo(1000000000)

# import numpy as np
# import matplotlib.pyplot as plt
# x=np.arange(0,np.pi,0.002)
# print(x)
# y=x*np.cos(x)
# print(y)
# plt.plot(x,y)
# print(max(y)*2)
# plt.show()

#求不是素数的数目
# import math
# def sieve(size):
#     sieve=[True]*size
#     sieve[0]=False
#     sieve[1]=False
#     for i in range(2,int(math.sqrt(size))+1):
#         k=i*2
#         while k<size:
#             sieve[k]=False
#             k+=i
#     return sum(1 for x in sieve if x)
# print(sieve(10))

# import base64
# import PIL
# from PIL import Image
# res = 'vOy67L3rveu667rsveu97b3rveq67L3qvey67L3rvey77brruuy667rquuy667zrvey97b3quuy66rzsu+u967rsu+u87L3tu+y96r3svey77Lzruuy66rrtuuu97Lvtuuu77b3svey67b3sveq67L3qveu67Lrsuuy6673rvOy77brsveu96r3suuu967rtveq67L3sveu9673quuy967rsveu767rsu+y967rsu+q967rsu+q86r3suu267b3ruu267Lvruuy67Lvtveu6673ruuu967rruuu97brru+y/6b/ovui67rrqve277L3suuu86r3tuuu6673svey967rsuuy66rzsuuy97Lrsveu67brruu286Lzpvui+6L7ovui+6L7ovui+6b3rveu87Lvrveu87Lvtu+u967vrvOy67Lvruuu9673svey967rtu+267Lrovui+6L7hp/OSvsmZ0JzDguqpoOi+6L7suuy6673su+y97b3uuuy867rsvey667vsuuu6673suuu967vsuOi+6Pm354MucwIlbAxPD2Q/aAJGAAiR5sa+6L3uvey967vtu+u67Lrruuy97Lrsu+u97b3suuy66bzsu+y96L7oillAFUUEVzR9KntUEU0kR3RUH0gjRnUuKc6+6bvsuuy6673suu267Lrsuuy67Lrtuuu97b3tuuq97bvsvei7i0cXVzpzKQtYF0YZWQRIM3sdTDRsL2QzdQBNjOi+7L3su+y87bvrveu87Lrsuu277Lrruuy67L3qu+y66L/AJx5MP2glBUcfdSNFEHEYQTJvMnkwYD1tOmzHayPVvui967rsuuy7673suuy97b3uu+u67Lvru+y67L3rvOypdkIEcSJ7VxZwK3UhdRx0M2c4lTxsNWnFbzaTxZLGaIHovu277bvsve267Lrrve277bvsuu2967rsveq67L76/SRLN2grF04RdSt/K2Q0Yz9pL382ajWVy5rCYMCZ0YzPyb7su+277L3suu2967ztuOy667rtuuu97Lrsve2//8UhYD4EdxxADkQfdyxjwYbUkDtuwJrNazyTyWo6hdmP27ev6Lrsuuu67Lrsuu297bruu+y97Lrru+u7673rvvDbVmVQHE4OWQ5DHH0yYcafxJc8lsmc0IXWj9WRN2vWte2P6+S+67jtuuu67Lrtvey77bvsveu77L3sveu67b3olSkGWnwueSgJQCthO27Nhj6VPmnBbcSA2rTbmOKIz4rwu9Oqvui67Lrsuuu967rsu+277Lrsuuy67Lrsuu296MU/CS9vO3EmAkAuazie1oQ2YTdtwo7lmsuK56b6vd2K4bnmj6Louuy47rrrvey667vuuO267L3tuuy77bvtvtpjKHIndDlzIAd3N2nMtdCXN2zFju+9157cu/Wg8qH6r9y0946C6LzuuOy77Lrsveu77rvsuuy767rruu696b6aX1RyI3IjfkUleTSc4YvKYjNlxLDSm+mi877yrvit8KT7uPmw7/6+7rvsvey97brsu+677Lvrvey67rzsu+i+c3NXciZvIQ5/Ikg1iOOexX7BZC6f1NyKpOOw/KTurIuo/KH2veDYuuq77brsveq97bvvu+y67L3tuOy87LvonTp6L3cnbiIQdRZIyoHXjMNkPZbAkPTU4Ircoei28aKD0eyk9L7eoK3ouu667brsvey47rrrvO267brvtui+6OkEcC9/VXpQIHUWec+Z2r/AZyeG27jqj+aI2rXvpIDTgNbtooC53rSi6Lvsveu67brsuO+77Lrsuuu977fovuk2B24rc10KTxxCBpbZmdqNNnZ9f9e14aPfmOmpgMSNqfyo/K/9puCNmOi67bzru+y67Ljuu+y97LrsveunoejTGCQEVnInfi0ORhuc3I3ruN+L1biQ1ITa7qb+qI/Q/6L51/rV97/ejJPouuy77bvsveu577vtvey67b7vrYgvZw5RAHTEgoryrOaz4r3ovui+6L7ovui/5LPhp9mRqP7ShK361PW/2ISS6LvtuOy67L3su+647brsveu97L7YwW4WZY7LpP6/6KvmtPis+ajNkcX2rpfBmcik5L7grPOF0oavgKP1p9Voh+i67rrtu+y87bjuuuu67Lruvui+4/uD7sjcZOjIsrMnh/ei2rXQdjBrNIqHx5nJnfCJkdHPpsf/1oGp+bDZYInove297brsvey57rvrvO277LPzms6cqLHyA3HJq7PN9aKO2dRsx5zNiZH2m8qgyZ7z+NnTkZ/1hquA0Pm4356e6Lnsuu2667rtue+47bvuv+6b3pr+nua2+qnx8pmS76vwkNbkfMGM3MGr9aD2ncmh9qPgsbr7wJfQhNf7qd2Asei47bruuuy77bvuuu277b3rqvOx46bnt/S15aTXg+eY86rnoJI2oJbzn/eU3pHAo/Wc/6eI+MaUxo2qgdbPubzru+267brtu+y47bvtuuu96bTnt+Ol4tu2qcfmZvDFKm6A5ajYj/WnyaHP+d+UwKPxo+X+T/3CjNmN1P+i0cq+77nvu+277LrsuO6467rtveu85rP8sM19taOOBBfdvW9mwGP2/qb1nven94bclc6k/KThqWDrwYPbj92Cvtzmvu277brsuuu67bjuu+u67bvsvuSw57ptbY6Buj1q1U8mdQpN7v+iyJ7ynqjtuYzNpPag77Sa38aC0JHDl6GV6Lrtu+297brru+y47rjtvey67Lzsn+6cBTHX6KWZwzBYLGHGi/jIpcii+4KE17+NyJ3Ctej5Y+3aitf4rP+mr+i57rntvey97brtuO247r3quu666KbM1D7fkAZr62/GmObdi9qd86T3k8H/iNetlM2X5a6u35v2ypDX+LjfkpbovuC47bjsuuy97bjuuuy77L3tuO2+75eT547koplmQVM2jfbRh9uZx/q73LTp35nCo+7srpWflP+c24Kq9bWT/qrouu+57Lvsuu2477vsu+y67Lvvu+i+08fxsfeZn2tdKGo/ejSB/NX/suGk/qCKwKv7ktzYrrH8pM6Qx4uo75Lf7b/uu+677L3tuO267Lrsu+277LvuvqVRb8RvJk0PQzKYy34/vv7f/IPw1vmoj8qo+4qfh+Ou+bfmltbzveSIwMi+77jtu+267Ljsuuy67Lztu+y77L6lQzV0PQFAB0MznsFj0qKHwva9gtf80or1nNTs2bfku+mrx/ql7rjssD6rvuy77Lrtuuy57brsu+297Lvtuuu+p11MKkkMVQ1HKGrBmuerit+A3/nT+6GKypbSmOS665y2xq+Eqfm85bHUnq3ou+247brtue647rrsuO677brsvtZovPuo7WwcWR9vw47v1Y7S+K71o/bdkM+Y8Ljoo4wrlvWv5bzuvuSI1GGG6Lrtu+y67bjtuu2467vtu+247L7YfGougoGIKF4RbMyJ992Pq/Wph6P23JDApeystCNr5bHTnuKv74nWt/OJ/Oi+6LvruO657L3tuO277LvtuOu+1gRk1aTZcwktEm7Qv/rajKH10orSgtOa5K6sLGfrtzaV0rLqgdK9/9CFsuPlqeO+6rvtue267b3suOy47bvuvvoWWsiYIFoAWRpq3KT8o/yu8quK3YzxpaMwZOO5zGI9mdicPon83IW75qPoiMKZpui/7rntuO277Lvtu+64677kKRp5Rg1fBCgDctul77X0oIDclPytqi9cxbTMkMWXPm4yj4r2g77es+q14IHUazrDvui47rrru+267Lvuu+i+vGwxFk4IVXQ4fULEt9C25qaL/7TGwUcxj9BlOo3IYiJl8/+o2YGg963ggua+6r/RZ/rouO297Lvsuuu47brsvqlSLC1+GiFzLQV8LmI8nOvHqvzcTjGaz5c0YMaNNHDby6rD94LBbsqaOGs7ZMWE4Ys2prvtu+277bjuu++47b7GTkU7ewgyficTYBdmzYr/yZSZJoHMbz1pPpzNYzS5o8vWcgImZToKWBNCFE8WSiqS15657rvtu+267bjuuO6+x2BHKEdmPwBfLHcnh9S8gqbUmdCfOm7DmtOSKoqcwfV7ajdkLAQsCWQndSR8LmM3bT6euOu77Lrtu+247bvsvsFtVCArcC8EcSVOxIfapvWP24zPkcuY1IHLjpDygZ8bOmE9aD11LBmTNksZdSV4OGw9a7jtuuy97bvtuu247b7nEFcgV35eGnsmYNu69L3itt2A0YLfiOW+4rzkQXNWcA1kPWcwZjgqkjR/FEMneidzK2+57rrsu+y97Lvtu+6+7cMMHUEEfTRxxbf3oumK147agdWA5K7aYCEkWzh+Rmc0aTpoMVw0JpE1fidEIZbEaTN5uO+767vtuuy47rrruuigLXJ0Lmkxe9Sm6I3VhtCD14HStP2afjRlIHwvGlRhO2k4dCZjLjRsNnIecxtwwobbjbvuuuu77brsuu667rvqvsLpsD9yI5TjjNmYzIbXgNaF4qg3N2omcSVnUgwxbSZsPHdVbF04Zy9zGUksSSdgPJi47brsuu247bvuu+u47rrovuD0n9mM2pjNh9KC1oDTjP5nWDppP3EycCtoMmw8aT9tV35MLXUhSh1CK2YlYz17uO677bvuu+y77rvtu+257r7oveKam9a11ITYjN2O2tc8Dmw2XDxlMGo2aiRyOnQgeCwXexx2KGctfzFuMGHIlbntu+y67rruuu247L3tuO+57r7o/27is92Z04DXhfuFWzlkDGMxZQxoUnQjei19KQgvGHAgYjtqPZTBk8OUzoe47brtuu267rvsuO297rjuu+266Jdr2ILXnceeybPiOmk6XTdjC1svKkhjPAMhelJ/LCh+LpXHlseYzIXVncyAue2967rtuu277bvsu+247rjtuOiufjufyI3NktaiGQhvCV8KUjYVkzklVi92OnkhbEU6az2b1Y7djd+N2YTZt7jsuuy97LjsuO247LvvuO647rvrvdU2a8qk+2I0nG0/XwhbBnNg2ZIAB14haTpzIABJPIzOhdOz77jrtNqJ5o257bvsu+277Lvuuu647rjuue677b7nMpPzx/J4y1dTN18PUCnQu8JQVgpwI3ZSejIJk8qK6IXMt+i47LbbsvCVue6767vsvey77rjtu+2577jvuOi4sW0zF9swUAUEWQpVAxOL9ZgBCmYifVUEK2EAK6bQnu6AyIjlvfGO1Y3Mb7juu+y67Lvtu+267bvtuO+57r7NlphLEE1JeQ5ICl4GWHfgpN5RWzh9J3shZjdcV9atz5Tvu9GK4L31jT9kE3u97LrsuO297Lvtuuu77rngtuiz1KNCajJmNVs0YAlSNy+w5bsgGmMoCixiNWUJaH/pot2b3KHpj+Kx+bHFW2V3uO277Lvtuu247bvsu++577jsgfvuNCU3XzRnNV4Lbmzot+hgTg50RxE4XTJnIwuT8LjYm9C997TptPGv+aXESA=='
# encrypted = base64.standard_b64decode(res)
# print(len(encrypted))
# img = Image.frombytes('L', (64, 64), encrypted)
# img.show()

# a=np.array([[1,2,3],
#             [4,5,6],
#             [7,8,9],
#             [10,11,12]])
# print(np.linalg.matrix_rank(a))

# vector1=np.array([1,2,0,2,0])
# vector2=np.array([0,3,0,1,3])
# vector3=np.array([0,2,0,2,1])
#
# op12=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
# op13=np.dot(vector1,vector3)/(np.linalg.norm(vector1)*(np.linalg.norm(vector3)))
# op23=np.dot(vector2,vector3)/(np.linalg.norm(vector2)*(np.linalg.norm(vector3)))
# print(op12,op13,op23)

import cv2
import time
import datetime
import os
import socket
import threading
#import easygui

skt=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
skt.bind(('192.168.43.138',1989))
skt.listen()
skt2=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
skt2.bind(('192.168.43.138',1990))
skt2.listen()

strlist=[]
strlist2=[]
datalist=[]

def get_data():
    while True:
        try:
            # conn表示接受的数据流，addr表示客户端的地址
            conn, addr = skt.accept()
            # 接受客户端发送消息并打印
            msg = conn.recv(1024)
            strone=msg.decode('utf-8')
            if len(strlist)==5000:
                strlist.clear()
            strlist.append(strone)
            #print(float(strone[0:5]),float(strone[6:11]),float(strone[12:17]))
            if len(datalist)==6000:
                datalist.clear()
            # datalist.append(float(strone[0:5]))
            # datalist.append(float(strone[6:11]))
            # datalist.append(float(strone[12:17]))
            try:
                datalist.append(float(strone[0:5]))
                try:
                    datalist.append(float(strone[6:11]))
                    try:
                        datalist.append(float(strone[12:17]))
                    except ValueError:
                        continue
                except ValueError:
                    continue
            except ValueError:
                continue
            finally:
                conn.close()
        except:
            #skt.close()
            break

def get_data2():
    while True:
        try:
            # conn表示接受的数据流，addr表示客户端的地址
            conn, addr = skt2.accept()
            # 接受客户端发送消息并打印
            msg = conn.recv(1024)
            strone=msg.decode('utf-8')
            #print(float(strone[0:5]),float(strone[6:11]),float(strone[12:17]))
            if len(strlist2)==5000:
                strlist2.clear()
            strlist2.append(strone)
            conn.close()
        except:
            #skt2.close()
            break

# session=tf.Session()
# new_saver=tf.train.import_meta_graph('models/re.ckpt.meta')
# new_saver.restore(session,'models/re.ckpt')
# graph=tf.get_default_graph()
# y=tf.get_collection('pred_network')[0]
# X=graph.get_operation_by_name('X').outputs[0]

t = threading.Thread(target=get_data)
t.setDaemon(True)
t.start()

t2=threading.Thread(target=get_data2)
t2.setDaemon(True)
t2.start()
# 选取摄像头，0为笔记本内置的摄像头，1,2···为外接的摄像头
camera = cv2.VideoCapture(0)
flag=0#0表示平地，1表示楼道
if (camera.isOpened()):
    print('Open')
else:
    print('摄像头未打开')

#title = easygui.msgbox(msg="将于5s后开始记录摄像头移动情况！""\n""请离开保证背景稳定""\n"
            #           , title="运动检测追踪拍照", ok_button="开始执行")
#msg = easygui.msgbox(msg="移动物体保存于D:\\CCTVlook")
time.sleep(5)
background = None  # 初始化背景
pre_center_x=0
pre_center_y=0

while True:
    text = "No Target"
    flat = 0

    (grabbed, frame) = camera.read()

    # 对帧进行预处理，先转灰度图，再进行高斯滤波。
    # 用高斯滤波对图像处理，避免亮度、震动等参数微小变化影响效果
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray
        continue
    # 当前帧和第一帧的不同它可以把两幅图的差的绝对值输出到另一幅图上面来
    frameDelta = cv2.absdiff(background, gray)
    # 二值化
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # 腐蚀膨胀
    thresh = cv2.dilate(thresh, None, iterations=2)
    # 取轮廓
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    # 遍历轮廓
    CA_max=0
    center_x=0
    center_y=0
    max_cnt=None
    for c in cnts:
        if cv2.contourArea(c) <1000:  # 对于较小矩形区域，选择忽略
            continue
        flat = 1  # 设置一个标签，当有运动的时候为1
        # 计算轮廓的边界框，在当前帧中画出该框
        # (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if cv2.contourArea(c) > CA_max:
            CA_max=cv2.contourArea(c)
            max_cnt=c
        text = "Find Target! saved"
        print("Find Target!")

    (x, y, w, h) = cv2.boundingRect(max_cnt)
    center_x=x+w/2
    center_y=y+h/2

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #print(cv2.contourArea(max_cnt))
    if len(datalist)>300:

        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('models/re.ckpt.meta')
            new_saver.restore(sess, 'models/re.ckpt')
            graph = tf.get_default_graph()
            y = tf.get_collection('pred_network')[0]
            X = graph.get_operation_by_name('X').outputs[0]
            #
            # a = np.loadtxt("test.txt", delimiter=',', usecols=[3, 4, 5])
            # b = np.array(a)
            # b.resize([1, 1, 100, 3])
            trainlist=np.array(datalist[-300:])
            trainlist.resize([1,1,100,3])
            # print(trainlist)
            # re=session.run(y,feed_dict={X:trainlist})[0]
            # print(sess.run(y,feed_dict={X:a})[0])
            re=(sess.run(y, feed_dict={X: trainlist}))[0].tolist()
            reindex=re.index(max(re))
            print((sess.run(y, feed_dict={X: trainlist})))

            result = 0
            pre_result=0
            if pre_center_x != 0 and pre_center_y != 0:
                if abs(pre_center_x - center_x) < 4 and abs(pre_center_y - center_y) < 4:
                    if reindex == 4 and re[2]>re[3]:
                        result = 1
                    elif reindex==4 and re[0]<re[5]:
                        result = 2
                elif abs(pre_center_x - center_x)>15 or abs(pre_center_x - center_x)>15:
                    if reindex==1:
                        result = 4
                    elif reindex==4:
                        result = 3
            if result==0 and pre_result!=0:
                result=pre_result
            if result == 1:
                print("SIT")
                cv2.putText(frame, "SIT", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif result == 2:
                print("STAND")
                cv2.putText(frame, "STAND", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif result == 3:
                print("WALK")
                cv2.putText(frame, "WALK", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif result == 4:
                print("JOG")
                cv2.putText(frame, "JOG", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            pre_result=result




    # if pre_center_x != 0 and pre_center_y != 0:
    #     print(abs(pre_center_x - center_x),abs(pre_center_y - center_y))
    pre_center_y=center_y
    pre_center_x=center_x
    cv2.putText(frame, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    if strlist!=[]:
        cv2.putText(frame,"Acc Sensor: "+strlist[-1],(10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame,"The acc sensor is off",(10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if strlist2!=[]:
        cv2.putText(frame,"Heartrate: "+strlist2[-1],(10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame,"The heartrate sensor is off",(10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.imshow("Frame Delta", frameDelta)

    cv2.imshow("fps", frame)
    cv2.imshow("back",background)
    # cv2.imshow("Thresh", thresh)

    key = cv2.waitKey(1) & 0xFF

    # 如果q键被按下，跳出循环
    ch = cv2.waitKey(1)
    if key == ord("q"):
        break

    if flat == 1:  # 设置一个标签，当有运动的时候为1
        #fn = 'D:\CCTVlook\shot%d.jpg' % (shot_idx)
        #cv2.imwrite(fn, frame)
        continue

# session.close()
camera.release()
skt.close()
skt2.close()
cv2.destroyAllWindows()


