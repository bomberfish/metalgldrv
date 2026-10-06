#!/bin/bash
# Package glcts (built for iOS by `setup.sh --ios`) with metalgldrv into an app,
# sign it and install it on the connected device. run.sh --ios calls this.
#
#   CTS_DIR / CTS_BUILD_IOS   CTS checkout / iOS build dir
#   MGL_DEVICE, METALGLDRV_PROFILE, MGL_IDENTITY   as for scripts/ios-run.sh
#   CTS_BUNDLE_ID             default ca.bomberfish.glcts
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CTS_DIR="${CTS_DIR:-$HOME/metalgldrv-deps/cts}"
CTS_BUILD_IOS="${CTS_BUILD_IOS:-$CTS_DIR/build-ios}"
DRIVER_BUILD="$REPO/build-ios"
DEVICE="${MGL_DEVICE:-00008142-001538391E80C01C}"
PROFILE="${METALGLDRV_PROFILE:-$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles/d069a67f-97c7-453a-8012-9ed70ad39456.mobileprovision}"
IDENTITY="${MGL_IDENTITY:-Apple Development: hariz@bomberfish.ca (6W8DD89D7N)}"
BUNDLE_ID="${CTS_BUNDLE_ID:-ca.bomberfish.glcts}"
MODULES="$CTS_BUILD_IOS/external/openglcts/modules"
APP="$CTS_BUILD_IOS/glcts.app"
export PATH="/opt/homebrew/bin:$PATH"

[[ -x "$MODULES/glcts.app/glcts" ]] || { echo "glcts for iOS not built; run tests/cts/setup.sh --ios" >&2; exit 2; }
if [[ ! -f "$DRIVER_BUILD/build.ninja" ]]; then
    cmake -G Ninja -B "$DRIVER_BUILD" -S "$REPO" -DCMAKE_TOOLCHAIN_FILE="$REPO/cmake/ios.cmake" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
fi
ninja -C "$DRIVER_BUILD" metalgldrv >/dev/null

rm -rf "$APP"
mkdir -p "$APP/Frameworks"
cp "$MODULES/glcts.app/glcts" "$APP/glcts"
cp "$DRIVER_BUILD/libmetalgldrv.dylib" "$APP/Frameworks/"
# Test data (the mustpass case lists stay on the host; chunks are copied over per launch)
rsync -a --exclude 'data/mustpass' "$MODULES/gl_cts" "$APP/"
rsync -a "$MODULES/gles2" "$APP/"
cp "$PROFILE" "$APP/embedded.mobileprovision"
cat > "$APP/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>glcts</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleName</key><string>glcts</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
  <key>MinimumOSVersion</key><string>26.0</string>
  <key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>
  <key>UILaunchScreen</key><dict/>
  <key>UIRequiresFullScreen</key><true/>
  <key>UIFileSharingEnabled</key><true/>
  <key>UISupportedInterfaceOrientations</key><array><string>UIInterfaceOrientationLandscapeLeft</string><string>UIInterfaceOrientationLandscapeRight</string><string>UIInterfaceOrientationPortrait</string></array>
  <key>UIApplicationSceneManifest</key><dict>
    <key>UIApplicationSupportsMultipleScenes</key><false/>
    <key>UISceneConfigurations</key><dict/>
  </dict>
</dict></plist>
PLIST
TMP="$CTS_BUILD_IOS/sign"
mkdir -p "$TMP"
security cms -D -i "$PROFILE" > "$TMP/profile.plist"
TEAM=$(/usr/libexec/PlistBuddy -c "Print :TeamIdentifier:0" "$TMP/profile.plist")
/usr/libexec/PlistBuddy -x -c "Print :Entitlements" "$TMP/profile.plist" > "$TMP/ent.plist"
/usr/libexec/PlistBuddy -c "Set :application-identifier $TEAM.$BUNDLE_ID" "$TMP/ent.plist"
/usr/libexec/PlistBuddy -c "Delete :keychain-access-groups" "$TMP/ent.plist" 2>/dev/null || true
codesign -f -s "$IDENTITY" "$APP/Frameworks/libmetalgldrv.dylib" >/dev/null
codesign -f -s "$IDENTITY" --entitlements "$TMP/ent.plist" "$APP" >/dev/null
xcrun devicectl device install app --device "$DEVICE" "$APP" >/dev/null
echo "[ios] installed $BUNDLE_ID on $DEVICE"
