/****************************************************************************
 *
 * Terminal Guidance - Tracking Control Panel for QGroundControl
 *
 ****************************************************************************/

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import QGroundControl
import QGroundControl.Controls
import QGroundControl.ScreenTools
import QGroundControl.Palette

Rectangle {
    id: root
    width: panelColumn.width + (_margins * 2)
    height: panelColumn.height + (_margins * 2)
    color: qgcPal.window
    radius: ScreenTools.defaultFontPixelWidth * 0.5
    border.color: qgcPal.windowShade
    border.width: 1

    // Connection config
    property string serverHost: "192.168.1.147"
    property int serverPort: 5000
    property string serverUrl: "http://" + serverHost + ":" + serverPort
    property bool autoConnect: true

    property real _margins: ScreenTools.defaultFontPixelWidth
    property real _panelWidth: ScreenTools.defaultFontPixelWidth * 24

    // State
    property bool connected: false
    property bool connecting: false
    property string lastError: ""
    property string trackingState: "searching"
    property bool controlEnabled: false
    property int lockedTargetId: -1
    property var targets: []
    property real fps: 0
    property real inferenceMs: 0
    property bool showConfig: !connected

    QGCPalette { id: qgcPal }

    Timer {
        id: pollTimer
        interval: 500
        running: root.visible && (connected || autoConnect)
        repeat: true
        onTriggered: fetchStatus()
    }

    Component.onCompleted: if (autoConnect) fetchStatus()

    function fetchStatus() {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                connecting = false
                if (xhr.status === 200) {
                    try {
                        var r = JSON.parse(xhr.responseText)
                        connected = true
                        lastError = ""
                        trackingState = r.tracking_state || "searching"
                        controlEnabled = r.control_enabled || false
                        lockedTargetId = r.locked_target_id !== null ? r.locked_target_id : -1
                        targets = r.targets || []
                        fps = r.fps || 0
                        inferenceMs = r.inference_ms || 0
                    } catch (e) {
                        setError("Parse error")
                    }
                } else {
                    setError(xhr.status === 0 ? "No response" : "HTTP " + xhr.status)
                }
            }
        }
        xhr.onerror = function() { connecting = false; setError("Network error") }
        xhr.ontimeout = function() { connecting = false; setError("Timeout") }
        try {
            serverUrl = "http://" + serverHost + ":" + serverPort
            xhr.open("GET", serverUrl + "/api/status")
            xhr.timeout = 2000
            xhr.send()
        } catch (e) {
            connecting = false
            setError("Failed")
        }
    }

    function setError(msg) {
        connected = false
        lastError = msg
    }

    function sendCommand(endpoint, body) {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                fetchStatus()
            }
        }
        xhr.open("POST", serverUrl + endpoint)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.timeout = 3000
        xhr.send(body ? JSON.stringify(body) : "")
    }

    function doConnect() {
        autoConnect = true
        connecting = true
        lastError = ""
        fetchStatus()
    }

    function doDisconnect() {
        autoConnect = false
        connected = false
        connecting = false
        pollTimer.stop()
    }

    ColumnLayout {
        id: panelColumn
        anchors.centerIn: parent
        spacing: _margins * 0.75
        width: _panelWidth

        // Header
        RowLayout {
            Layout.fillWidth: true
            spacing: _margins * 0.5

            QGCLabel {
                text: "Terminal Guidance"
                font.pointSize: ScreenTools.mediumFontPointSize
                font.bold: true
                Layout.fillWidth: true
            }

            // Connection indicator (clickable)
            Rectangle {
                width: ScreenTools.defaultFontPixelWidth * 1.2
                height: width
                radius: width / 2
                color: connecting ? qgcPal.colorOrange : (connected ? qgcPal.colorGreen : qgcPal.colorRed)

                SequentialAnimation on opacity {
                    running: connecting
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.4; duration: 300 }
                    NumberAnimation { to: 1.0; duration: 300 }
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: showConfig = !showConfig
                    cursorShape: Qt.PointingHandCursor
                }
            }
        }

        // Connection config (shown when disconnected or toggled)
        Rectangle {
            Layout.fillWidth: true
            height: showConfig ? configCol.height + _margins : 0
            color: Qt.darker(qgcPal.window, 1.05)
            radius: 4
            clip: true
            visible: height > 0

            Behavior on height { NumberAnimation { duration: 120 } }

            ColumnLayout {
                id: configCol
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: _margins * 0.5
                spacing: _margins * 0.5

                // Server input - single line
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    QGCTextField {
                        id: hostInput
                        Layout.fillWidth: true
                        text: serverHost
                        placeholderText: "IP Address"
                        font.pointSize: ScreenTools.smallFontPointSize
                        onTextChanged: serverHost = text
                    }

                    QGCLabel {
                        text: ":"
                        font.pointSize: ScreenTools.smallFontPointSize
                    }

                    QGCTextField {
                        id: portInput
                        Layout.preferredWidth: ScreenTools.defaultFontPixelWidth * 6
                        text: serverPort.toString()
                        font.pointSize: ScreenTools.smallFontPointSize
                        validator: IntValidator { bottom: 1; top: 65535 }
                        onTextChanged: {
                            var p = parseInt(text)
                            if (p > 0 && p <= 65535) serverPort = p
                        }
                    }
                }

                // Connect/Disconnect
                RowLayout {
                    Layout.fillWidth: true
                    spacing: _margins * 0.5

                    QGCButton {
                        text: connected ? "Reconnect" : "Connect"
                        Layout.fillWidth: true
                        enabled: !connecting
                        onClicked: doConnect()
                    }

                    QGCButton {
                        text: "Disconnect"
                        Layout.fillWidth: true
                        enabled: connected || connecting
                        onClicked: doDisconnect()
                    }
                }

                // Status line
                QGCLabel {
                    Layout.fillWidth: true
                    font.pointSize: ScreenTools.smallFontPointSize
                    color: connected ? qgcPal.colorGreen : (lastError ? qgcPal.colorOrange : qgcPal.text)
                    text: {
                        if (connecting) return "Connecting..."
                        if (connected) return "Connected - " + fps.toFixed(0) + " FPS"
                        if (lastError) return lastError
                        return "Not connected"
                    }
                }
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: qgcPal.windowShade
        }

        // Main content - only when connected
        ColumnLayout {
            Layout.fillWidth: true
            spacing: _margins * 0.5
            visible: connected

            // Status grid
            GridLayout {
                columns: 2
                columnSpacing: _margins
                rowSpacing: 2
                Layout.fillWidth: true

                QGCLabel { text: "State:"; font.pointSize: ScreenTools.smallFontPointSize }
                QGCLabel {
                    text: trackingState.charAt(0).toUpperCase() + trackingState.slice(1)
                    font.bold: true
                    font.pointSize: ScreenTools.smallFontPointSize
                    color: trackingState === "locked" ? qgcPal.colorGreen :
                           trackingState === "acquiring" ? qgcPal.colorOrange :
                           trackingState === "lost" ? qgcPal.colorRed : qgcPal.text
                }

                QGCLabel { text: "Target:"; font.pointSize: ScreenTools.smallFontPointSize }
                QGCLabel {
                    text: lockedTargetId >= 0 ? "Locked #" + lockedTargetId : (targets.length + " detected")
                    font.pointSize: ScreenTools.smallFontPointSize
                    color: lockedTargetId >= 0 ? qgcPal.colorGreen : qgcPal.text
                }
            }

            // Lock controls
            RowLayout {
                Layout.fillWidth: true
                spacing: _margins * 0.5

                QGCButton {
                    text: "Lock"
                    Layout.fillWidth: true
                    enabled: lockedTargetId < 0 && targets.length > 0
                    onClicked: sendCommand("/api/tracking/lock", null)
                }

                QGCButton {
                    text: "Unlock"
                    Layout.fillWidth: true
                    enabled: lockedTargetId >= 0
                    onClicked: sendCommand("/api/tracking/unlock", null)
                }
            }

            // Target buttons (max 3)
            Repeater {
                model: targets.length > 0 && targets.length <= 3 ? targets : []

                QGCButton {
                    Layout.fillWidth: true
                    text: modelData["class"] + " (" + (modelData.confidence * 100).toFixed(0) + "%)"
                    enabled: lockedTargetId !== modelData.id
                    highlighted: lockedTargetId === modelData.id
                    font.pointSize: ScreenTools.smallFontPointSize
                    onClicked: sendCommand("/api/tracking/lock", {"target_id": modelData.id})
                }
            }

            // Separator
            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: qgcPal.windowShade
            }

            // Control buttons
            RowLayout {
                Layout.fillWidth: true
                spacing: _margins * 0.5

                QGCButton {
                    id: enableBtn
                    text: controlEnabled ? "ACTIVE" : "Enable"
                    Layout.fillWidth: true
                    enabled: !controlEnabled && lockedTargetId >= 0

                    background: Rectangle {
                        color: controlEnabled ? qgcPal.colorGreen :
                               (enableBtn.enabled ? (enableBtn.pressed ? Qt.darker(qgcPal.colorGreen, 1.2) : qgcPal.colorGreen) : qgcPal.windowShade)
                        radius: 4
                        opacity: enableBtn.enabled || controlEnabled ? 1.0 : 0.5
                    }
                    contentItem: Text {
                        text: enableBtn.text
                        color: controlEnabled || enableBtn.enabled ? "white" : qgcPal.text
                        font.bold: true
                        font.pointSize: ScreenTools.defaultFontPointSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }

                    onClicked: sendCommand("/api/tracking/enable", null)
                }

                QGCButton {
                    text: "Disable"
                    Layout.fillWidth: true
                    enabled: controlEnabled
                    onClicked: sendCommand("/api/tracking/disable", null)
                }
            }

            // Emergency Stop
            QGCButton {
                Layout.fillWidth: true
                text: "EMERGENCY STOP"

                background: Rectangle {
                    color: parent.pressed ? Qt.darker("#cc0000", 1.2) : "#cc0000"
                    radius: 4
                }
                contentItem: Text {
                    text: "EMERGENCY STOP"
                    color: "white"
                    font.bold: true
                    font.pointSize: ScreenTools.defaultFontPointSize
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                onClicked: sendCommand("/api/emergency-stop", null)
            }
        }

        // Disconnected state
        QGCLabel {
            visible: !connected && !showConfig
            Layout.fillWidth: true
            text: "Tap indicator to configure"
            font.pointSize: ScreenTools.smallFontPointSize
            color: qgcPal.colorGrey
            horizontalAlignment: Text.AlignHCenter
        }
    }
}
