import 'package:flutter/material.dart';

class DrawingPoint {
  final double x;
  final double y;
  final double pressure;

  DrawingPoint({required this.x, required this.y, this.pressure = 1.0});

  Offset toOffset(Size size) => Offset(x * size.width, y * size.height);
}

class DrawingStroke {
  final List<DrawingPoint> points;
  final Color color;
  final double thickness;
  final bool glow;

  DrawingStroke({
    required this.points,
    required this.color,
    required this.thickness,
    this.glow = true,
  });

  // Simple bounds check for grabbing
  bool contains(Offset p, Size size, double tolerance) {
    for (final pt in points) {
      final opt = pt.toOffset(size);
      if ((opt - p).distance < tolerance + thickness) return true;
    }
    return false;
  }
}

class DrawingController extends ChangeNotifier {
  List<DrawingStroke> _strokes = [];
  final List<List<DrawingStroke>> _undoStack = [];
  final List<List<DrawingStroke>> _redoStack = [];

  List<DrawingStroke> get strokes => _strokes;

  void addStroke(DrawingStroke s) {
    _saveState();
    _strokes.add(s);
    notifyListeners();
  }

  void updateLastStroke(DrawingStroke s) {
    if (_strokes.isNotEmpty) {
      _strokes[_strokes.length - 1] = s;
      notifyListeners();
    }
  }

  void removeStroke(DrawingStroke s) {
    _saveState();
    _strokes.remove(s);
    notifyListeners();
  }

  void clear() {
    _saveState();
    _strokes = [];
    notifyListeners();
  }

  void undo() {
    if (_undoStack.isNotEmpty) {
      _redoStack.add(List.from(_strokes));
      _strokes = _undoStack.removeLast();
      notifyListeners();
    }
  }

  void redo() {
    if (_redoStack.isNotEmpty) {
      _undoStack.add(List.from(_strokes));
      _strokes = _redoStack.removeLast();
      notifyListeners();
    }
  }

  void notify() => notifyListeners();

  void _saveState() {
    _undoStack.add(List.from(_strokes));
    _redoStack.clear();
    if (_undoStack.length > 50) _undoStack.removeAt(0);
  }
}
