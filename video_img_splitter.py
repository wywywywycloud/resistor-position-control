import cv2

sensor_num = 1

while sensor_num <= 16:

    video_path = 'C:/Users/wywycloud/PycharmProjects/course-resistor-position-control/resources/sensor_videos/type_1/'
    video_name = 'vid (' + str(sensor_num) + ').MOV'
    cap = cv2.VideoCapture(video_path + video_name)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    i = 0
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imwrite('resources/temp/frame{:d}.jpg'.format(count), frame)
            count += 15 # i.e. at 30 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            cv2.imwrite(video_path + str(sensor_num) + '/Frame' + str(i) + '.jpg', frame)
            i += 1

        else:
            cap.release()
            break

    sensor_num += 1


cap.release()
cv2.destroyAllWindows()
