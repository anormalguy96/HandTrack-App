package com.boxscraps.handtrack_engine

import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import android.graphics.YuvImage
import android.graphics.Rect

internal fun ImageProxy.toBitmap(): Bitmap {
  val yuvBytes = yuv420ToNv21(this)
  val yuv = YuvImage(yuvBytes, ImageFormat.NV21, width, height, null)
  val out = ByteArrayOutputStream()
  yuv.compressToJpeg(Rect(0, 0, width, height), 80, out)
  val bytes = out.toByteArray()
  return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

private fun yuv420ToNv21(image: ImageProxy): ByteArray {
  val yBuffer = image.planes[0].buffer
  val uBuffer = image.planes[1].buffer
  val vBuffer = image.planes[2].buffer

  val ySize = yBuffer.remaining()
  val uSize = uBuffer.remaining()
  val vSize = vBuffer.remaining()

  val nv21 = ByteArray(ySize + uSize + vSize)

  yBuffer.get(nv21, 0, ySize)
  vBuffer.get(nv21, ySize, vSize)
  uBuffer.get(nv21, ySize + vSize, uSize)

  return nv21
}
