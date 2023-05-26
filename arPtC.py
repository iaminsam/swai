###########################################################################
#                                                               
# Auto Drive Car  V0.9
#
# $ python3 ar.py <이미지 디렉터리>
#
# Example:
# $ python3 ar.py                디렉토리: data
# $ python3 ar.py video          디렉토리: video
#
# 
# 학습 파일(*.h5)이 없어도 화면 부팅은 하여 차량 수동 조작은 가능 하도록 한다.
#
# FEB 2 2022
#
# SAMPLE Electronics co.                                        
# http://www.ArduinoPLUS.cc                                     
#                                                               
###########################################################################
hornDist = 350            # 경적이 울리는 거리
stopDist = 110            # 차량과 벽과의 거리 (차량 정지, 자율 주행과 수동 전진에서 해당)
runSpeed = 70             # 60, 70, 80, 90  차량 주행 속도(speed)
angleGain = 0.8           # 0.7, 0.8, 0.9, 1.0, 1.1  조향각 이득(gain)
#--------------------------------------------------------------------------
videoDir = 'data'         # 기본 디렉토리 이름
# modelName = '_model_final.h5'
modelName = '_model_check.pt'
modelFile = None          # *_model_final.h5
#--------------------------------------------------------------------------
RED     =   0,  0,255     # Red
GREEN   =   0,255,  0     # Green
BLUE    = 255,  0,  0     # Blue
PINK    = 255,  0,255     # Pink
MAGENTA = 255,255,  0     # Magenta(Sky Blue)
YELLOW  =   0,255,255     # Yellow
WHITE   = 255,255,255     # White
BLACK   =   0,  0,  0     # Black
GRAY    =  86, 86, 86     # Gray
#--------------------------------------------------------------------------
import VL53L0X             # ToF Ranger
import tsC as ts           # Traffic Signal Parameter Set
import WS2812
WS2812.rainbow(1, 80, 1)
#--------------------------------------------------------------------------
import os
import sys
import time
import pickle
import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO


# import tensorflow as tf
# from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
#--------------------------------------------------------------------------
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()

        # elu=Expenential Linear Unit, similar to leaky Relu
        # skipping 1st hiddel layer (nomralization layer), as we have normalized the data

        # Convolution Layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3)),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ELU(inplace=True)
        )

        # Fully Connected Layers
        self.layer2 = nn.Sequential(
            # nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features=18 * 64, out_features=100),
            nn.ELU(inplace=True),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(inplace=True),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(inplace=True)
        )

        # Output Layer
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.shape[0], -1)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
#--------------------------------------------------------------------------
WIN_YU = 310
WIN_YD = 390
WIN_XL = 100
WIN_XR = 540
viewWin = np.zeros((400,640,3),np.uint8)  # 표시되는 윈도우 가로, 세로, 컬러층, 8비트
#--------------------------------------------------------------------------
START_VIEW_Y = 80                     # 출력 프레임 Y 시작 (신호등 설정과 공유)
TIME_VIEW = 90                        # 검출 영역 표시 시간 (신호등 설정과 공유)
#--------------------------------------------------------------------------
MOTOR_L_PWM = 12                      # GPIO.12    왼쪽 모터 펄스폭 변조
MOTOR_L_DIR = 5                       # GPIO.5     원쪽 모터 방향
MOTOR_R_PWM = 13                      # GPIO.13    오른쪽 모터 펄스폭 변조
MOTOR_R_DIR = 6                       # GPIO.6     오른쪽 모터 방향
BUZZER = 23                           # GPIO.23    경적
MUSIC = 24                            # GPIO.24    후진 사운드
LAMP_R_YELLOW = 20                    # Right Yellow Lamp
LAMP_L_YELLOW = 26                    # Left Yellow Lamp
LAMP_BRAKE = 21                       # 브레이크 Lamp
LIGHT_SENSOR = 25
#--------------------------------------------------------------------------
GPIO.setwarnings(False)               # GPIO 관련 경고 메시지 출력 금지
GPIO.setmode(GPIO.BCM)                # BCM 핀 번호
GPIO.setup(MOTOR_L_PWM,GPIO.OUT)      # 왼쪽 모터 펄스폭
GPIO.setup(MOTOR_L_DIR,GPIO.OUT)      # 왼쪽 모터 방향
GPIO.setup(MOTOR_R_PWM,GPIO.OUT)      # 오른쪽 모터 펄스폭
GPIO.setup(MOTOR_R_DIR,GPIO.OUT)      # 오른쪽 모터 방향
GPIO.setup(BUZZER,GPIO.OUT)           # 경적 
GPIO.setup(MUSIC,GPIO.OUT)            # 후진 사운드 
GPIO.setup(LAMP_R_YELLOW,GPIO.OUT)    # 우회전 깜빡이 램프
GPIO.setup(LAMP_L_YELLOW,GPIO.OUT)    # 좌회전 깜빡이 램프
GPIO.setup(LAMP_BRAKE,GPIO.OUT)       # 브레이크 램프
GPIO.setup(LIGHT_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#--------------------------------------------------------------------------
MOTOR_L = GPIO.PWM(MOTOR_L_PWM,500)   # 왼쪽 모터 PWM(펄스폭 변조) 주파수 500Hz
MOTOR_R = GPIO.PWM(MOTOR_R_PWM,500)   # 오른쪽 모터 PWM(펄스폭 변조) 주파수 500Hz
MOTOR_L.start(0)                      # 왼쪽 모터 PWM(펄스폭 변조) 값 0 으로 시작
MOTOR_R.start(0)                      # 오른쪽 모터 PWM(펄스폭 변조) 값 0 으로 시작
#--------------------------------------------------------------------------
RESTART_NUM = 33
MODEL_FILE = False
tSign = RESTART_NUM
model = None 
menu = 'View  LAmp  Horn  [HOME]:auto-Run  [SPACE]:stop  [ESC]:exit'
autoRun = False             # 자동차 동작 상태 (True:자율주행 False:정지)
mL = 0; mR = 0              # 모터 속도
tSign = 0                   # 적색 신호가 감지 됬을 때 0 보다 큰 수로 설정, 0 이면 주행 아니면 정지 
preKey = ord(' ')
# Motor Run Function ------------------------------------------------------
def motorRun(leftMotor, rightMotor):
# +N = Forward
#  0 = Stop
# -N = Backward

    # PWM 값이 99보다 크면 99로 조정
    amotorL = abs(leftMotor) 
    if amotorL>99: amotorL = 99
    amotorR = abs(rightMotor)
    if amotorR>99: amotorR = 99
    
    if leftMotor >= 0:
        GPIO.output(MOTOR_L_DIR,GPIO.HIGH)     # 왼쪽 모터 전진
    else:
        GPIO.output(MOTOR_L_DIR,GPIO.LOW)      # 왼쪽 모터 후진
    MOTOR_L.ChangeDutyCycle(amotorL)           # 왼쪽 모터 PWM 값 설정

    if rightMotor >= 0:
        GPIO.output(MOTOR_R_DIR,GPIO.HIGH)     # 오른쪽 모터 전진
    else:
        GPIO.output(MOTOR_R_DIR,GPIO.LOW)      # 오른쪽 모터 후진
    MOTOR_R.ChangeDutyCycle(amotorR)           # 오른쪽 모터 PWM 값 설정
# 자동차 운행 정지 ------------------------------------------------------------
def stopCar():
    global autoRun, tSign, preKey, BLINK_LEFT, BLINK_RIGHT, mL, mR

    autoRun = False
    tSign = RESTART_NUM
    preKey = ord(' ')
    BLINK_LEFT = False; BLINK_RIGHT = False
    GPIO.output(MUSIC,GPIO.LOW)         # 후진 사운드 멈춤
    mL = 0; mR = 0
#==========================================================================
def main():
    global tSign, autoRun, preKey, BLINK_LEFT, BLINK_RIGHT, mL, mR

    # 신호등 처리 라이브러리를 초기화 합니다.
    ts.videoDir = videoDir
    ts.tSigInit()
    # Create a VL53L0X object
    tof = VL53L0X.VL53L0X()
    # Start ranging
    tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
    camera=cv.VideoCapture(0,cv.CAP_V4L) # 카메라 객체 생성
    camera.set(3, 640)           # 카메라 비디오 X(가로) 크기
    camera.set(4, 480)           # 카메라 비디오 Y(세로) 크기
    angle = 0                    # 조향 각도
    timeCycle = 0                # 실행 주기 
    horn = 0                     # 경적 유지 시간
    YM = 220                     # 모터 바 그래프 중앙 위치
    barColor = YELLOW
    countNum = 0
    blinkCount = 0
    BLINK_LEFT = False
    BLINK_RIGHT = False
    HEAD_LIGHT_AUTO = False      # True:전조등 자동 점등   False:수동 점등
    HEAD_LIGHT_STATE = False     # True:전조등 켜짐       False:전조등 꺼짐
    cds = 'NIGHT'                # CdS 의 상태기록 Day:밝음, Night:어두움 
    hla = 'MANUAL'               # Head Light 동작모드 Auto:자동, Manual:수동 
    hls = ' '                    # Head Light 상태 On:켜짐, None:꺼짐
    WS2812.turnOnLamp(BLACK)     # 전조등 끄기
    stopCar()

    while(camera.isOpened()):
        startCycle = time.time()                        # 프레임 시작 시각
        countNum += 1                                   # 프레임 카운트

        # 빛 감지 센서 -------------------------------------------------------
        if GPIO.input(LIGHT_SENSOR):
            cds = 'NIGHT'
        else:
            cds = 'DAY'

        if hla == 'AUTO' and cds == 'NIGHT':
            WS2812.turnOnLamp(WHITE)  # 전조등 On
        elif hls == 'ON':
            WS2812.turnOnLamp(WHITE)  # 전조등 On
        elif hls == 'POLICE 1':
            WS2812.policeCar1(12, 7, int(countNum/3))  # Blue, Red, Number
        elif hls == 'POLICE 2':
            WS2812.policeCar2(12, 7, int(countNum/2))  # Blue, Red, Number
        elif hls == 'FIRE TRUCK':
            WS2812.fireTruck(12, int(countNum/2))
        elif hls == 'AMBULANCE':
            WS2812.ambulance(12, int(countNum/2))
        elif hls == 'CONSTRUCT':
            WS2812.construction(7, int(countNum/2))
        elif hls == ' ':
            WS2812.turnOnLamp(BLACK)  # 전조등 Off

        # 경적 -------------------------------------------------------------
        if horn:
            GPIO.output(BUZZER,GPIO.HIGH)
            horn -= 1                                   # 경적 유지 시간 감소
        else:
            GPIO.output(BUZZER,GPIO.LOW)                # 경적 끄기
        # 카메라로 부터 1 프레임 영상 가져오기
        _, frame = camera.read()
        #frame = cv.flip(frame,-1)    # 입력 이미지를 필요하면 상하 반전
        viewWin[0:400] = frame[80:480]
        image = viewWin[WIN_YU:WIN_YD,WIN_XL:WIN_XR]
        image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        image = cv.GaussianBlur(image, (3,3), 0)
        image = cv.resize(image, (200,66)) 
        procImg = image / 255         # 0-255 범위(정수)를 0.000-0.999 범위(유리수)로 변환
        # 신호등 인식 -------------------------------------------------------
        # 적색 신호등이 인식되면 tSign 에 non zero(RESTART_NUM) 를 저장한다.
        # tSign 이 0 이 아니면 차량은 정지하고 0 이되면 주행 시작한다.
        # tSign 은 1 프레임 에서 1 씩 감소하여 0 이 된다. 
        v = ts.trafficSign(viewWin, SET_MODE=False)  # 신호등 인식 함수 실행 
        if v == 'G': tSign = 0                      # 녹색 신호등
        if v == 'R': tSign = RESTART_NUM            # 적색 신호등
        if tSign > 0: tSign -= 1 # 적색 신호등 이후에 일정시간 지나면 녹색 신호등이 없어도 주행 시작
        if tSign > 0:            # 적색 신호등 소등 이후 카운트 표시
            cv.putText(viewWin,f'{tSign:2d}',(80,50),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        # 방향 지시등 --------------------------------------------------------
        blinkCount += 1
        if blinkCount > 10: blinkCount = 0
        if blinkCount > 5:
            if BLINK_RIGHT:
                GPIO.output(LAMP_R_YELLOW,GPIO.HIGH)    # 오른쪽 노란색
                cv.rectangle(viewWin,(600,330),(620,350),YELLOW,-1)
            if BLINK_LEFT:
                GPIO.output(LAMP_L_YELLOW,GPIO.HIGH)    # 왼쪽 노란색
                cv.rectangle(viewWin,(20,330),(40,350),YELLOW,-1)
        else:
            GPIO.output(LAMP_R_YELLOW,GPIO.LOW)         # 오른쪽 노란색 끄기
            GPIO.output(LAMP_L_YELLOW,GPIO.LOW)         # 왼쪽 노란색 끄기
        # 전방 거리 측정, 경적 발생, 차량 정지 -----------------------------------
        distance = tof.get_distance()      # VL53L0X TOF 거리 센서 측정 값
        if 0 < distance < 2000: 
            cv.putText(viewWin, f'{distance:3d} MM', (240,320),cv.FONT_HERSHEY_COMPLEX_SMALL,2,WHITE)
            # 경적 구간에 들어오면 스크린에 정지 표시하고 경적 소리 낸다.
            if distance < hornDist:
                cv.circle(viewWin, (320, 160), 100, RED, 25) #정지 표시
                cv.line(viewWin, (320+70, 160-70), (320-70, 160+70), RED, 25)
                GPIO.output(LAMP_BRAKE,GPIO.HIGH)           # 브레이크 적색 램프 점등
                if preKey == 80 or preKey == 82:            # 자율 주행 또는 전진시 경적
                    horn = True
                if distance < stopDist:
                    horn = False
                    if (preKey == 80 or preKey == 82 or preKey == ord('u')):
                        autoRun = False
                        tSign = RESTART_NUM
                        preKey = ord(' ')
                        mL = 0; mR = 0
                        BLINK_LEFT = True; BLINK_RIGHT = True
            else:
                GPIO.output(LAMP_BRAKE,GPIO.LOW)            # 브레이크 적색 램프 끄기

        viewWin[340:340+33, 270:270+100] = cv.pyrDown(image)
        cv.putText(viewWin, f'{timeCycle:4d} mS', (260,20),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv.putText(viewWin, menu, (20,393),cv.FONT_HERSHEY_PLAIN,1,YELLOW)
        cv.putText(viewWin, f'{hornDist} {stopDist}                              {hla} {cds} {hls}',
                      (20,373),cv.FONT_HERSHEY_PLAIN,1.2,YELLOW)
        if countNum < TIME_VIEW:
            cv.rectangle(viewWin,(WIN_XL+2,WIN_YU+2),(WIN_XR-2,WIN_YD-2),YELLOW,2)  # 차선 검색 영역 표시
            cv.putText(viewWin, 'Line Detect Area', (230,WIN_YU+20),cv.FONT_HERSHEY_COMPLEX_SMALL,1,YELLOW)
        # 자율 주행 모드 조향 각도 검출 -----------------------------------------
        if MODEL_FILE and autoRun:    # 학습 파일이 존재하고 자율주행 모드일 때 실행
            X = np.asarray([procImg])
            X = torch.Tensor(X).permute(0, 3, 1, 2)
            # angle = int((model.predict(X)[0])*angleGain)
            angle = 0
            model.eval()
            with torch.no_grad():
                angle = int(model(X)[0] * angleGain)

            cv.putText(viewWin, f'{angle:3d}', (180,260),cv.FONT_HERSHEY_COMPLEX_SMALL,6,WHITE)
            mL = runSpeed + angle     # 왼쪽 모터 속도 값(PWM)
            mR = runSpeed - angle     # 오른쪽 모터 속도 값(PWM) 
            cv.putText(viewWin, f'{mL:3d}', (30,225),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
            cv.putText(viewWin, f'{mR:3d}', (560,225),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        # 왼쪽 모터와 오른쪽 모터의 속도 크가 값(PWM)을 바 그래프로 표시 --------------
        if autoRun:
            barColor = GREEN
        else:
            barColor = YELLOW
        # 왼쪽 모터 바 그래프 -------------------------------------------------
        if mL >= 0:
            ys = YM - mL; ye = YM
        else:
            ys = YM; ye = YM - mL  
        cv.rectangle(viewWin,(20,YM-100),(30,YM+100),BLACK,-1)
        cv.rectangle(viewWin,(20,ys),(30,ye),barColor,-1)
        # 오른쪽 모터 바 그래프 -----------------------------------------------
        if mR >= 0:
            ys = YM - mR; ye = YM
        else:
            ys = YM; ye = YM - mR  
        cv.rectangle(viewWin,(610,YM-100),(620,YM+100),BLACK,-1)
        cv.rectangle(viewWin,(610,ys),(620,ye),barColor,-1)
        # 자율 주행 모드에서 Red 신호(0 보다 큰 수)일 때 정지 ----------------------
        if tSign > 0 and autoRun:   
            mL = 0; mR = 0
        motorRun(mL, mR)                      # 왼쪽 모터, 오른쪽 모터 동작
        # 깜빡이 설정 -------------------------------------------------------
        if preKey == ord('d') or preKey == 84:
            pass
        else:
            if abs(mL - mR) > 9:
                if (mL - mR) > 0:
                    BLINK_LEFT = False; BLINK_RIGHT = True
                else:
                    BLINK_LEFT = True; BLINK_RIGHT = False 
            else:
                BLINK_LEFT = False; BLINK_RIGHT = False
        # 윈도우 디스프레이 ---------------------------------------------------
        cv.imshow(modelFile, viewWin)
        #cv.moveWindow(modelFile, 80, 20)     # Pi 화면에서 Window 고정 위치 지정
        # 수동 조작 모드, 파라메터 설정 -----------------------------------------
        keyBoard = cv.waitKey(1)
        #print(hex(keyBoard))
        if keyBoard == 0x1B or keyBoard == 0x09 or keyBoard == ord('q'):    # ESC/TAB 프로그램 종료
            break                                    # ESC, TAB, Q
        elif keyBoard == 80 or keyBoard == ord('r'): # Home, R 자율 주행 시작
            autoRun = True; tSign = 0; preKey = 80
        elif keyBoard == ord('v'):    # 카메라, 신호등 영역 표시
            countNum = 0
            ts.countNum = 0
        elif keyBoard == ord('h'):    # 경적
            horn = 5                  # 5 프레임 동안 경적 울림
        elif keyBoard == ord('l'):    # 전조등 매뉴얼 모드
            if hls == ' ':
                hls = 'ON'
            elif hls == 'ON':
                hls = 'POLICE 1'
            elif hls == 'POLICE 1':
                hls = 'POLICE 2'
            elif hls == 'POLICE 2':
                hls = 'FIRE TRUCK'
            elif hls == 'FIRE TRUCK':
                hls = 'AMBULANCE'
            elif hls == 'AMBULANCE':
                hls = 'CONSTRUCT'
            elif hls == 'CONSTRUCT':
                hls = ' '
        elif keyBoard == ord('a'):    # 전조등 점등 모드
            if hla == 'MANUAL':
                hla = 'AUTO'          # 전조등 자동 점등 
            elif hla == 'AUTO':
                hla = 'MANUAL'        # 전조등 수동 점등
        # 수동 차량 조작 -----------------------------------------------------
        elif keyBoard == 82:                         # 전진 Arrow Up(82)
            if preKey == ord(' '):
                autoRun = False; tSign = 0; preKey = keyBoard
                mL = 80; mR = 80
            else:
                stopCar()
        elif keyBoard == 84:                         # 후진 Arrow Down(84)
            if preKey == ord(' '):
                autoRun = False; tSign = 0; preKey = keyBoard
                mL = -70; mR = -70
                GPIO.output(MUSIC,GPIO.HIGH)         # 후진 멜로디
                BLINK_LEFT = True; BLINK_RIGHT = True  # 비상등
            else:
                stopCar()
        elif keyBoard == 81:                         # 좌회전 Left(81)
            if preKey == ord(' '):
                autoRun = False; tSign = 0; preKey = keyBoard
                mL = -70; mR = 70
            else:
                stopCar()
        elif keyBoard == 83:                         # 우회전 Right(83)
            if preKey == ord(' '):
                autoRun = False; tSign = 0; preKey = keyBoard
                mL = 70; mR = -70
            else:
                stopCar()
        elif keyBoard == ord(' ') or keyBoard == 87: # Space, End 정지
            stopCar()

        endCycle = time.time()                          # 프레임 종료 시각
        timeCycle = int((endCycle - startCycle)*1000)   # 프레임 소요 시간
    #----------------------------------------------------------------------
    motorRun(0, 0)                    # 모터 정지
    WS2812.turnOnLamp(BLACK)          # 전조등 소등
    GPIO.cleanup()                    # GPIO 모듈의 점유 리소스 해제
    tof.stop_ranging()                # Stop ranging
    cv.destroyAllWindows()            # 열려진 모든 윈도우 닫기
#==========================================================================
if __name__ == '__main__':

    print('\n')
    print('Python Version:     ', sys.version)
    print('OpenCV Version:     ', cv.__version__)
    print('Pytorch Version:    ', torch.__version__)
    print('\n')

    if len(sys.argv) >= 2:
        videoDir = sys.argv[1]         # Video 저장 디렉토리
    
    print('videoDir', videoDir)
    modelFile = './'+videoDir+'/'+videoDir+modelName
    # 학습 파일 
    if os.path.exists(modelFile): 
        # model = load_model(modelFile)
        model = NvidiaModel()
        model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))
        MODEL_FILE = True
        print('\n학습 모델 파일 ', modelFile, ' 을 읽었습니다.\n')
    else:
        print('\n학습 모델 파일 ', modelFile, ' 이 없습니다.\n')
        modelFile = 'Manual Mode'
    # 트랙 윈도우 파일
    t = './'+videoDir+'/'+videoDir+'_track.pickle'
    if os.path.exists(t):
        with open(t, 'rb') as f:
            d = pickle.load(f)
            WIN_XL = d[0]; WIN_XR = d[1]; WIN_YU = d[2]; WIN_YD = d[3]
        print('트랙 윈도우 파일', t, ' 을 읽었습니다.')
    else:
        print('트랙 윈도우 파일', t, ' 이 없습니다.')

    main()

##########################################################################
