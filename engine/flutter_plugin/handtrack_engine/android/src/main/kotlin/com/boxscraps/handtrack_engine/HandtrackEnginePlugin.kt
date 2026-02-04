package com.boxscraps.handtrack_engine

import android.content.Context
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.platform.PlatformViewRegistry

class HandtrackEnginePlugin : FlutterPlugin {

  override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    register(binding.applicationContext, binding.binaryMessenger, binding.platformViewRegistry)
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {}

  private fun register(ctx: Context, messenger: BinaryMessenger, registry: PlatformViewRegistry) {
    registry.registerViewFactory(
      "handtrack_engine/view",
      HandtrackViewFactory(ctx, messenger)
    )
  }
}
