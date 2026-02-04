import time
import cv2

def main():
    cap = cv2.VideoCapture(0)
    last = time.time()
    ema = 0.0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        dt = max(1e-6, now - last)
        inst = 1.0 / dt
        ema = inst if ema == 0 else (ema * 0.9 + inst * 0.1)
        last = now
        n += 1

        cv2.putText(frame, f"Capture FPS: {ema:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("bench_webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Frames:", n)

if __name__ == "__main__":
    main()
