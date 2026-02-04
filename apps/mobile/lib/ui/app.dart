import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

class HandTrackApp extends StatelessWidget {
  const HandTrackApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'HandTrack',
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF185E50),
      ),
      home: const HomeScreen(),
    );
  }
}
