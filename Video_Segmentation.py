import cv2
import os

def waitKey(delay = False, fps=100): # Define wait key
    if delay == True:
        key = cv2.waitKey(int(1000 / fps)) & 0xff # Case 1 (without button operations)
    else:
        key = cv2.waitKey(100) & 0xff # Case 2 (With button operations)
    return key

def segmentation(source_path, target_path=""): # Function for video segmentation with input and target path
    if target_path=="": # Error handling/ Exception handling: if target path is not set, proceed with default path
        file_path, file_name_w_ext = os.path.split(source_path)
        filename, file_extension = os.path.splitext(file_name_w_ext)
        target_file = filename + '_processed.mp4'
        target_path = file_path + '/' + target_file

    backSub = cv2.createBackgroundSubtractorMOG2() # Background removal function (Gaussian detection)
    backSub.setDetectShadows(True) # High gray intensity pixels near foreground will be shadow

    cap = cv2.VideoCapture(source_path) # Video capture: extract video from input path
    fwidth = int(cap.get(3)) # Extract frame width and height for output video resolution (Imp, save video)
    fheight = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Default video saving format
    out = cv2.VideoWriter(target_path, fourcc, 20, (fwidth, fheight))
    while True:

        ret, frame = cap.read() # extract current frame
        if frame is None:
            break

        fgMask = backSub.apply(frame,learningRate=0.01) # Extract image/ frame mask

        final_frame = cv2.bitwise_and(frame, frame, mask=fgMask) # Apply mask on current frame
        out.write(final_frame) # Save video frame by frame
        cv2.imshow('Processed_Video', final_frame) # Show output frame

        #cv2.imshow('Frame1', frame) # show original frame
        #cv2.imshow('Frame2', fgMask) # show mask

        key = waitKey()
        if key == ord('q'):
            break

    # Release members
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Process complete!")

if __name__ == "__main__": # Main function to test above function

    source_path = ".../folder1/video_1.mp4" # Change input path
    target_path = ".../folder1/video_1_processed.mp4" # Change target path

    segmentation(source_path) # Test function: Add target path if required as second function argument
