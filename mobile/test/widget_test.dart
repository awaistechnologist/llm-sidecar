import 'package:flutter_test/flutter_test.dart';
import 'package:sidecar_app/main.dart';

void main() {
  test('normalise turns whatever was pasted into an origin', () {
    expect(normalise('https://mac.tail1.ts.net:8443/app/'), 'https://mac.tail1.ts.net:8443');
    expect(normalise('mac.tail1.ts.net:8443'), 'https://mac.tail1.ts.net:8443');
    expect(normalise('  https://mac.tail1.ts.net/app/?x=1 '), 'https://mac.tail1.ts.net');
    expect(normalise('http://localhost:4001'), 'http://localhost:4001');
  });
}
