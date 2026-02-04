import 'package:flutter/material.dart';
import '../../core/api_client.dart';
import '../../core/ads_client.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _api = ApiClient();
  final _ads = AdsClient();

  Map<String, dynamic>? _health;
  Map<String, dynamic>? _adsConfig;
  String? _error;

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final h = await _api.getHealth();
      final a = await _ads.getAdsConfig();
      setState(() {
        _health = h;
        _adsConfig = a;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('HandTrack')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Gesture Canvas (stub)',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
            const SizedBox(height: 10),
            Container(
              height: 220,
              width: double.infinity,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.black12),
              ),
              child: const Center(
                child: Text(
                  'Camera + Hand Tracking Engine goes here\n(Android: MediaPipe/TFLite, iOS: CoreML)',
                  textAlign: TextAlign.center,
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                ElevatedButton(onPressed: _load, child: const Text('Refresh')),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(_error ?? '',
                      style: const TextStyle(color: Colors.red),
                      overflow: TextOverflow.ellipsis),
                )
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView(
                children: [
                  _Card(title: 'API /health', data: _health),
                  _Card(title: 'Ads config (optional)', data: _adsConfig),
                  const SizedBox(height: 12),
                  const Text('Next (MVP)',
                      style: TextStyle(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 6),
                  const Text('• Implement platform plugin: capture → inference → render'),
                  const Text('• Add adaptive quality (resolution, ROI, frame skip)'),
                  const Text('• Add export (9:16 templates)'),
                  const Text('• Add consent UI (ads + analytics)'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Card extends StatelessWidget {
  final String title;
  final Map<String, dynamic>? data;

  const _Card({required this.title, required this.data});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          Text(
            data == null ? 'Loading…' : data.toString(),
            style: const TextStyle(fontFamily: 'monospace'),
          ),
        ],
      ),
    );
  }
}
