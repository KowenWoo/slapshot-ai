import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart';
import 'package:permission_handler/permission_handler.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: CameraExample(),
    );
  }
}

class CameraExample extends StatefulWidget {
  @override
  _CameraExampleState createState() => _CameraExampleState();
}

class _CameraExampleState extends State<CameraExample> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isRecording = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    // Request camera and storage permissions
    await [
      Permission.camera,
      Permission.storage,
    ].request();

    _cameras = await availableCameras();
    if (_cameras!.isNotEmpty) {
      _controller = CameraController(_cameras![0], ResolutionPreset.high);
      await _controller!.initialize();
      if (!mounted) return;
      setState(() {});
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _startRecording() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }

    final directory = await getApplicationDocumentsDirectory();
    final filePath = join(directory.path, '${DateTime.now()}.mp4');

    try {
      await _controller!.startVideoRecording();
      setState(() {
        _isRecording = true;
      });
    } catch (e) {
      print(e);
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;

    try {
      final video = await _controller!.stopVideoRecording();
      setState(() {
        _isRecording = false;
      });
      print('Video saved to ${video.path}');
    } catch (e) {
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(
          title: Text('Camera Example'),
        ),
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Camera Example'),
      ),
      body: Column(
        children: <Widget>[
          Expanded(
            child: CameraPreview(_controller!),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: <Widget>[
                ElevatedButton(
                  onPressed: _isRecording ? null : _startRecording,
                  child: Text('Record'),
                ),
                ElevatedButton(
                  onPressed: _isRecording ? _stopRecording : null,
                  child: Text('Stop'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
