class FpsMeter {
  double _ema = 0;
  int _lastMs = 0;

  double tick(int nowMs) {
    if (_lastMs != 0) {
      final dt = (nowMs - _lastMs).clamp(1, 10000);
      final inst = 1000.0 / dt;
      _ema = _ema == 0 ? inst : (_ema * 0.9 + inst * 0.1);
    }
    _lastMs = nowMs;
    return _ema;
  }

  double get ema => _ema;
}
