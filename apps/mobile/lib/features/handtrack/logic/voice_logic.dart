import 'package:speech_to_text/speech_to_text.dart';
import 'package:flutter_tts/flutter_tts.dart';

class VoiceController {
  final SpeechToText _speech = SpeechToText();
  final FlutterTts _tts = FlutterTts();

  bool _isListening = false;
  String _lastWords = "";
  Function(String)? onCommand;
  Function(String)? onStatus;

  bool get isListening => _isListening;

  Future<void> init() async {
    await _speech.initialize(
      onStatus: (s) => onStatus?.call("Voice: $s"),
      onError: (e) => onStatus?.call("Voice Error: ${e.errorMsg}"),
    );
    await _tts.setLanguage("en-US");
    await _tts.setSpeechRate(0.5);
  }

  Future<void> speak(String text) async {
    await _tts.speak(text);
  }

  Future<void> startListening() async {
    if (!_isListening) {
      bool available = await _speech.initialize();
      if (available) {
        _isListening = true;
        _speech.listen(
          onResult: (val) {
            _lastWords = val.recognizedWords;
            if (val.finalResult) {
              onCommand?.call(_lastWords);
              _isListening = false;
            }
          },
        );
      }
    }
  }

  Future<void> stopListening() async {
    if (_isListening) {
      await _speech.stop();
      _isListening = false;
    }
  }
}
