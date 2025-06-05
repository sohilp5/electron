#import <Cocoa/Cocoa.h>
#import <node_api.h> // For N-API

// Helper function to get the NSWindow from the Buffer handle
NSWindow* GetWindowFromElectronHandle(napi_env env, napi_value bufferValue) {
    void* bufferData;
    size_t bufferLength;
    if (napi_get_buffer_info(env, bufferValue, &bufferData, &bufferLength) != napi_ok) {
        napi_throw_error(env, NULL, "[NativeModule] Failed to get buffer info from window handle.");
        return nil;
    }

    // Electron's getNativeWindowHandle() on macOS returns a pointer to the NSView
    // that is the BrowserWindow's content view.
    NSView* mainContentView = (__bridge NSView*)bufferData;
    if (!mainContentView || ![mainContentView isKindOfClass:[NSView class]]) {
        napi_throw_error(env, NULL, "[NativeModule] Invalid NSView handle provided.");
        return nil;
    }
    return [mainContentView window];
}

// N-API exposed function
napi_value SetEnhancedWindowProperties(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_status status = napi_get_cb_info(env, info, &argc, argv, NULL, NULL);

    if (status != napi_ok || argc < 1) {
        napi_throw_type_error(env, NULL, "[NativeModule] Invalid arguments: Expected window handle.");
        return NULL;
    }

    NSWindow* window = GetWindowFromElectronHandle(env, argv[0]);
    if (window == nil) {
        // Error already thrown by GetWindowFromElectronHandle or invalid handle
        napi_throw_error(env, NULL, "[NativeModule] Could not retrieve NSWindow from handle.");
        return NULL;
    }

    // 1. Set Sharing Type to None (Primary goal for invisibility)
    // This tells the system the window's content is private and should not be shared or recorded.
    if ([window respondsToSelector:@selector(setSharingType:)]) {
        [window setSharingType:NSWindowSharingNone];
        NSLog(@"[NativeModule] Window sharingType set to NSWindowSharingNone.");
    } else {
        NSLog(@"[NativeModule] Warning: NSWindow does not respond to setSharingType. (macOS version too old?)");
    }

    // 2. Set Window Collection Behavior
    // This influences how the window interacts with Mission Control, App Switcher, Spaces, etc.
    NSWindowCollectionBehavior currentBehavior = [window collectionBehavior];
    NSWindowCollectionBehavior newBehavior = currentBehavior;

    newBehavior |= NSWindowCollectionBehaviorCanJoinAllSpaces;    // Allow on all Spaces
    newBehavior |= NSWindowCollectionBehaviorStationary;           // Window doesn't move with Spaces during Mission Control
    newBehavior |= NSWindowCollectionBehaviorIgnoresCycle;        // Hide from Command-Tab application switcher
    newBehavior |= NSWindowCollectionBehaviorFullScreenAuxiliary; // Behaves as an auxiliary window in full screen

    // Remove standard "managed" behavior if you want more control (can have side effects)
    // newBehavior &= ~NSWindowCollectionBehaviorManaged;
    // newBehavior &= ~NSWindowCollectionBehaviorParticipatesInCycle; // Alternative to IgnoresCycle

    [window setCollectionBehavior:newBehavior];
    NSLog(@"[NativeModule] Window collectionBehavior updated.");

    // You can also log the final behavior for debugging:
    // NSLog(@"[NativeModule] Final collectionBehavior: %lu", (unsigned long)[window collectionBehavior]);

    napi_value napi_true;
    napi_get_boolean(env, true, &napi_true);
    return napi_true;
}

// Module registration
napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc = {
        "setEnhancedWindowProperties",
        NULL,
        SetEnhancedWindowProperties,
        NULL,
        NULL,
        NULL,
        napi_default,
        NULL
    };
    napi_status status = napi_define_properties(env, exports, 1, &desc);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "[NativeModule] Failed to define native function.");
    }
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)