#!/bin/bash
# Build the iOS driver + test runner, sign, install on the connected device and
# run with console output. Usage: scripts/ios-run.sh [test-filter...]
# Env: MGL_DEVICE (UDID), MGL_PROFILE (provisioning profile path), MGL_IDENTITY
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-ios"
DEVICE="${MGL_DEVICE:-00008142-001538391E80C01C}"
PROFILE="${MGL_PROFILE:-$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles/d069a67f-97c7-453a-8012-9ed70ad39456.mobileprovision}"
IDENTITY="${MGL_IDENTITY:-Apple Development: hariz@bomberfish.ca (6W8DD89D7N)}"
BUNDLE_ID="${MGL_BUNDLE_ID:-ca.bomberfish.mglrunner}"
APP="$BUILD/MGLRunner.app"

if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -G Ninja -B "$BUILD" -S "$ROOT" -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/ios.cmake" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
fi
ninja -C "$BUILD" metalgldrv mgltest

rm -rf "$APP"
mkdir -p "$APP/Frameworks"
cp "$BUILD/tests/mgltest.app/mgltest" "$APP/MGLRunner"
cp "$BUILD/libmetalgldrv.dylib" "$APP/Frameworks/"
cp "$PROFILE" "$APP/embedded.mobileprovision"
cat > "$APP/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>MGLRunner</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleName</key><string>MGLRunner</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
  <key>MinimumOSVersion</key><string>26.0</string>
  <key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>
  <key>UILaunchScreen</key><dict/>
  <key>UIRequiresFullScreen</key><true/>
  <key>UISupportedInterfaceOrientations</key><array><string>UIInterfaceOrientationLandscapeLeft</string><string>UIInterfaceOrientationLandscapeRight</string><string>UIInterfaceOrientationPortrait</string></array>
  <key>UIApplicationSupportsIndirectInputEvents</key><true/>
  <key>UIApplicationSceneManifest</key><dict>
    <key>UIApplicationSupportsMultipleScenes</key><false/>
    <key>UISceneConfigurations</key><dict/>
  </dict>
</dict></plist>
PLIST
TMP="$BUILD/sign"
mkdir -p "$TMP"
security cms -D -i "$PROFILE" > "$TMP/profile.plist"
TEAM=$(/usr/libexec/PlistBuddy -c "Print :TeamIdentifier:0" "$TMP/profile.plist")
/usr/libexec/PlistBuddy -x -c "Print :Entitlements" "$TMP/profile.plist" > "$TMP/ent.plist"
/usr/libexec/PlistBuddy -c "Set :application-identifier $TEAM.$BUNDLE_ID" "$TMP/ent.plist"
/usr/libexec/PlistBuddy -c "Delete :keychain-access-groups" "$TMP/ent.plist" 2>/dev/null || true
codesign -f -s "$IDENTITY" "$APP/Frameworks/libmetalgldrv.dylib" >/dev/null
codesign -f -s "$IDENTITY" --entitlements "$TMP/ent.plist" "$APP" >/dev/null
xcrun devicectl device install app --device "$DEVICE" "$APP" >/dev/null
echo "== running on device $DEVICE"
xcrun devicectl device process launch --device "$DEVICE" --console --terminate-existing \
  --environment-variables '{"MGL_LOG":"2"}' "$BUNDLE_ID" "$@" 2>&1 | grep -v "^Waiting\|^Launched\|^$"
