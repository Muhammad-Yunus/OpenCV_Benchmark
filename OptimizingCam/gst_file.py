def gst_file_loader(filename, fs=30):
    return 'filesrc location=%s ! \
        qtdemux ! queue ! h264parse ! \
        omxh264dec ! nvvidconv ! \
        video/x-raw,format=BGRx,framerate=%s/1 ! \
        queue ! videoconvert ! queue ! \
        video/x-raw, format=BGR ! appsink' % (filename, fs)