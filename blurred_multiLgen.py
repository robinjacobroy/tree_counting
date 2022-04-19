import numpy as np
import cv2 as cv
import sys

masks = []
seeds = []

name = sys.argv[1]
class App():
    
    
    def update(self, dummy=None):
        
        if self.seed_pt is None:
            cv.namedWindow('hsv_filter', cv.WINDOW_NORMAL)
            
            cv.imshow('hsv_filter', self.orig)
            return
        
        hsv = self.img.copy()
        rgb = self.orig.copy()
        #self.mask[:] = 0
        
        l_h = cv.getTrackbarPos("L - H", "hsv_filter")
        l_s = cv.getTrackbarPos("L - S", "hsv_filter")
        l_v = cv.getTrackbarPos("L - V", "hsv_filter")
        u_h = cv.getTrackbarPos("U - H", "hsv_filter")
        u_s = cv.getTrackbarPos("U - S", "hsv_filter")
        u_v = cv.getTrackbarPos("U - V", "hsv_filter")
        
        if self.seed_pt[0]<hsv.shape[1]:
            h,s,v = hsv[self.seed_pt[::-1]]
            lower_hsv = np.array([abs(h-l_h), abs(s-l_s), abs(v-l_v)])
            self.l_hsv = lower_hsv.copy()
            upper_hsv = np.array([h+u_h, s+u_s, v+u_v])
            self.u_hsv = upper_hsv.copy()
            #try:
            self.mask = cv.inRange(hsv, lower_hsv, upper_hsv)
            result = cv.bitwise_and(rgb, rgb, mask=self.mask)
            cv.circle(result, self.seed_pt, 5, (0, 0, 255), -1)
            cv.circle(self.orig, self.seed_pt, 5, (0, 0, 255), -1)
            final = cv.hconcat([self.img,result])
            cv.imshow('hsv_filter', final)
        
        
        
    
    def onmouse(self, event, x, y, flags, param):
        if flags & cv.EVENT_FLAG_LBUTTON:
            self.seed_pt = x, y
            
            self.update()
            
            
    def run(self):
        
        self.orig = cv.imread(name)[:4000,:4000]
        self.orig = cv.GaussianBlur(self.orig, (25,25), 80)
        self.img = cv.cvtColor(self.orig, cv.COLOR_BGR2HSV)
        if self.img is None:
            print('Failed to load image file:')
            sys.exit(1)

        h, w = self.img.shape[:2]
        self.mask = np.zeros((h+2, w+2), np.uint8)
        self.seed_pt = None
        self.update()
        cv.setMouseCallback('hsv_filter', self.onmouse)
        cv.createTrackbar("L - H", "hsv_filter", 30, 100, self.update)
        cv.createTrackbar("L - S", "hsv_filter", 30, 100, self.update)
        cv.createTrackbar("L - V", "hsv_filter", 30, 100, self.update)
        cv.createTrackbar("U - H", "hsv_filter", 30, 100, self.update)
        cv.createTrackbar("U - S", "hsv_filter", 30, 100, self.update)
        cv.createTrackbar("U - V", "hsv_filter", 30, 100, self.update)
        while True:
            ch = cv.waitKey()
            if ch == 27:
                break
            elif ch == ord('s'):
                masks.append(self.mask)
                seeds.append((self.l_hsv, self.u_hsv))
                self.update()
                
        print("Done")
        
App().run()
cv.destroyAllWindows()



hsv = cv.imread(name)
hsv = cv.cvtColor(hsv, cv.COLOR_BGR2HSV)
frame = np.zeros(hsv.shape[:2])
for num, pt in enumerate(seeds):
    mask = cv.inRange(hsv, pt[0], pt[1])
    frame[mask==255] = num+1
cv.imwrite("multi_mask.png", frame)
