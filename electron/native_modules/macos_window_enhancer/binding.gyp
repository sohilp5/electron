{
  "targets": [
    {
      "target_name": "macos_window_enhancer",
      "sources": [ "macos_window_enhancer.m" ],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "MACOSX_DEPLOYMENT_TARGET": "10.13",
            "OTHER_CFLAGS": [
              "-std=c11", 
              "-Wall",
              "-Wextra",
              "-Wno-unused-parameter",
              "-fobjc-arc" 
            ],
            "OTHER_LDFLAGS": [
              "-framework Cocoa",
              "-framework Foundation"
            ]
          },
          "link_settings": {
            "libraries": [
              "-framework Cocoa",
              "-framework Foundation"
            ]
          }
        }]
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').targets\"):node_addon_api"
      ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ]
    }
  ]
}