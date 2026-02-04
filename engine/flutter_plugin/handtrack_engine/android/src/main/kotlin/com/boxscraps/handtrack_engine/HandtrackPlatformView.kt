package com.boxscraps.handtrack_engine

import android.annotation.SuppressLint
import android.content.Context
import android.os.SystemClock
import android.util.Size
import android.view.View
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import io.flutter.embedding.android.FlutterActivity
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class HandtrackPlatformView(
  private val appContext: Context,
  private val viewContext: Context,
  messenger: BinaryMessenger,
  private val viewId: Int,
  params: Map<String, Any>
) : PlatformView {

  private val previewView = PreviewView(viewContext)
  private val analyzerExecutor = Executors.newSingleThreadExecutor()
  private val running = AtomicBoolean(true)

  private var preferFront = (params["preferFrontCamera"] as? Boolean) ?: true
  private var enablePreview = (params["enablePreview"] as? Boolean) ?: true
  private var targetFps = ((params["targetFps"] as? Number) ?: 30).toInt()

  private var maxHands = ((params["maxHands"] as? Number) ?: 1).toInt()
  private var minDet = ((params["minDetection"] as? Number) ?: 0.5).toFloat()
  private var minPres = ((params["minPresence"] as? Number) ?: 0.5).toFloat()
  private var minTrk = ((params["minTracking"] as? Number) ?: 0.5).toFloat()

  private var handLandmarker: HandLandmarker? = null
  private var cameraProvider: ProcessCameraProvider? = null

  private var lastFrameTs = 0L
  private var fpsEma = 0.0
  private var lastResultTs = 0L

  private val eventChannel = EventChannel(messenger, "handtrack_engine/events_$viewId")
  private var eventSink: EventChannel.EventSink? = null

  private val methodChannel = MethodChannel(messenger, "handtrack_engine/methods_$viewId")

  init {
    previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

    eventChannel.setStreamHandler(object : EventChannel.StreamHandler {
      override fun onListen(arguments: Any?, events: EventChannel.EventSink?) { eventSink = events }
      override fun onCancel(arguments: Any?) { eventSink = null }
    })

    methodChannel.setMethodCallHandler { call, result ->
      when (call.method) {
        "setConfig" -> {
          @Suppress("UNCHECKED_CAST")
          val m = (call.arguments as? Map<String, Any>) ?: emptyMap()
          applyConfig(m)
          result.success(null)
        }
        "pause" -> { running.set(false); result.success(null) }
        "resume" -> { running.set(true); result.success(null) }
        "switchCamera" -> { preferFront = !preferFront; bindCamera(); result.success(null) }
        else -> result.notImplemented()
      }
    }

    setupLandmarker()
    bindCamera()
  }

  private fun applyConfig(m: Map<String, Any>) {
    targetFps = ((m["targetFps"] as? Number) ?: targetFps).toInt()
    maxHands = ((m["maxHands"] as? Number) ?: maxHands).toInt()
    minDet = ((m["minDetection"] as? Number) ?: minDet).toFloat()
    minPres = ((m["minPresence"] as? Number) ?: minPres).toFloat()
    minTrk = ((m["minTracking"] as? Number) ?: minTrk).toFloat()
    enablePreview = (m["enablePreview"] as? Boolean) ?: enablePreview
    setupLandmarker()
    bindCamera()
  }

  private fun setupLandmarker() {
    handLandmarker?.close()
    val base = BaseOptions.builder()
      // You must place hand_landmarker.task into: android/src/main/assets
      .setModelAssetPath("hand_landmarker.task")
      .build()

    val options = HandLandmarker.HandLandmarkerOptions.builder()
      .setBaseOptions(base)
      .setRunningMode(RunningMode.LIVE_STREAM)
      .setNumHands(maxHands)
      .setMinHandDetectionConfidence(minDet)
      .setMinHandPresenceConfidence(minPres)
      .setMinTrackingConfidence(minTrk)
      .setResultListener { res: HandLandmarkerResult, input ->
        publishResult(res, input.width, input.height)
      }
      .setErrorListener { e ->
        eventSink?.success(JSONObject(mapOf("error" to (e.message ?: "unknown"))).toString())
      }
      .build()

    handLandmarker = HandLandmarker.createFromOptions(appContext, options)
  }

  @SuppressLint("UnsafeOptInUsageError")
  private fun bindCamera() {
    val activity = (viewContext as? FlutterActivity) ?: return
    val lifecycleOwner = activity as LifecycleOwner

    val providerFuture = ProcessCameraProvider.getInstance(viewContext)
    providerFuture.addListener({
      cameraProvider = providerFuture.get()
      cameraProvider?.unbindAll()

      val selector = CameraSelector.Builder()
        .requireLensFacing(if (preferFront) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK)
        .build()

      val preview = Preview.Builder().build().apply {
        if (enablePreview) setSurfaceProvider(previewView.surfaceProvider)
      }

      val analysis = ImageAnalysis.Builder()
        .setTargetResolution(Size(640, 480))
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

      analysis.setAnalyzer(analyzerExecutor) { imageProxy ->
        try {
          if (!running.get()) { imageProxy.close(); return@setAnalyzer }
          throttleAndDetect(imageProxy)
        } catch (_: Throwable) {
          imageProxy.close()
        }
      }

      cameraProvider?.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
    }, ContextCompat.getMainExecutor(viewContext))
  }

  private fun throttleAndDetect(imageProxy: ImageProxy) {
    val now = SystemClock.uptimeMillis()

    if (lastFrameTs != 0L) {
      val dt = (now - lastFrameTs).coerceAtLeast(1)
      val inst = 1000.0 / dt.toDouble()
      fpsEma = if (fpsEma == 0.0) inst else (fpsEma * 0.9 + inst * 0.1)
    }
    lastFrameTs = now

    // FPS throttle (simple): skip frames if we’re above target
    val minInterval = (1000.0 / targetFps).toLong().coerceAtLeast(1)
    if (lastResultTs != 0L && (now - lastResultTs) < minInterval) {
      imageProxy.close()
      return
    }

    val bmp = imageProxy.toBitmap() // minimal conversion; optimize later with direct MPImage path
    val mpImage = BitmapImageBuilder(bmp).build()
    handLandmarker?.detectAsync(mpImage, now)
    imageProxy.close()
  }

  private fun publishResult(res: HandLandmarkerResult, imageW: Int, imageH: Int) {
    val now = SystemClock.uptimeMillis()
    lastResultTs = now

    val out = JSONObject()
    out.put("tsMs", now)
    out.put("imageW", imageW)
    out.put("imageH", imageH)
    out.put("fps", fpsEma)
    out.put("latencyMs", 0.0) // can be wired from timestamps later

    val handsArr = JSONArray()
    for (i in res.landmarks().indices) {
      val landmarks = res.landmarks()[i]
      val pts = JSONArray()
      for (p in landmarks) {
        val obj = JSONObject()
        obj.put("x", p.x())
        obj.put("y", p.y())
        obj.put("z", p.z())
        obj.put("conf", 1.0)
        pts.put(obj)
      }

      val handObj = JSONObject()
      handObj.put("handedness", res.handedness()[i][0].categoryName())
      handObj.put("score", res.handedness()[i][0].score())
      handObj.put("landmarks", pts)
      handsArr.put(handObj)
    }

    out.put("hands", handsArr)
    eventSink?.success(out.toString())
  }

  override fun getView(): View = previewView

  override fun dispose() {
    running.set(false)
    try { handLandmarker?.close() } catch (_: Throwable) {}
    try { cameraProvider?.unbindAll() } catch (_: Throwable) {}
    analyzerExecutor.shutdown()
  }
}
