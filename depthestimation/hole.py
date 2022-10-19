import cv2

def draw_hole(gray,img):
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    #image 가장자리 삭제
    margin =  0
    #crop_gray = gray[margin:height-margin, margin:width-margin]

    w = gray.shape[0]
    h = gray.shape[1]

    #image smooth
    #tt= cv2.resize(gray, dsize =(20,20))
   # gray= cv2.resize(tt, dsize =(h,w),interpolation=cv2.INTER_LINEAR)
    #cv2.imshow("tt",gray)
    #intergral image (평균계산 빠르게)
    isum= cv2.integral(gray)

    #평균낼 영역 최소 사이즈 (이미지 사이즈 기준)
    divide = 16
    base_size_w= w/divide
    base_size_h= h/divide

    #시작 위치
    start_x = 0
    start_y = 0 

    #이전 평균 낸 영역 사이즈 저장
    prev_w = w
    prev_h = h

    #기준 사이즈의 몇배로 시작할것인지 
    initial_step =8

    #output 
    last_pos_x = 0
    last_pos_y = 0
    
    for step in range(initial_step,1,-1):
        
        #평균낼 사이즈
        rect_width = int(base_size_w*step)
        rect_height = int(base_size_h*step)

        #검색 영역 제한
        end_x = start_x+(prev_w-rect_width)-1
        end_y = start_y+(prev_h-rect_height)-1
        
        minval = 100000000
        minx = -1
        miny = -1    
        
        #영역 평균 최소 탐색
        for x in range(start_x,end_x,3):
            for y in range(start_y,end_y,3):
                val = isum[x+rect_width,y+rect_height] - isum[x+rect_width,y] - isum[x,y+rect_height] +isum[x,y]            
                if val < minval:
                    minval = val
                    minx = x
                    miny = y

        start_x = minx
        start_y = miny
        prev_w = rect_width
        prev_h = rect_height

        #cv2.circle(img, (miny+margin+int(rect_width/2.0), minx+margin+int(rect_width/2.0)) , rect_width, (0,16*step,255-16*step),1+int(4/step))

        last_pos_x = minx+margin+int(rect_height/2.0)
        last_pos_y = miny+margin+int(rect_width/2.0)

    return last_pos_y, last_pos_x
    
    
