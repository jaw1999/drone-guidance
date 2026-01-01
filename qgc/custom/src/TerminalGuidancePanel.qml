/****************************************************************************
 *
 * Terminal Guidance - Tracking Control Panel for QGroundControl
 *
 * Minimal flight operations panel. All configuration is done via the
 * web UI on the companion computer (http://<pi-ip>:5000).
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

    // Connection to companion computer - editable in UI
    property string serverHost: "192.168.2.11"
    property int serverPort: 5000
    property string serverUrl: "http://" + serverHost + ":" + serverPort
    property bool autoConnect: false

    property real _margins: ScreenTools.defaultFontPixelWidth
    property real _panelWidth: ScreenTools.defaultFontPixelWidth * 22

    // Tracking state from API
    property bool connected: false
    property bool connecting: false
    property string lastError: ""
    property string trackingState: "searching"
    property bool controlEnabled: false
    property int lockedTargetId: -1
    property int targetCount: 0
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
                        targetCount = r.targets ? r.targets.length : 0
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
            setError("Request failed")
        }
    }

    function setError(msg) {
        connected = false
        lastError = msg
    }

    function sendCommand(endpoint, body) {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    fetchStatus()
                }
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
        spacing: _margins * 0.5
        width: _panelWidth

        // Header with connection indicator
        RowLayout {
            Layout.fillWidth: true
            spacing: _margins * 0.5

            QGCLabel {
                text: "Tracking"
                font.pointSize: ScreenTools.mediumFontPointSize
                font.bold: true
                Layout.fillWidth: true
            }

            // Connection indicator (tap to show/hide config)
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

        // Connection config panel (collapsible)
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
                spacing: _margins * 0.4

                // IP:Port input row
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    QGCTextField {
                        id: hostInput
                        Layout.fillWidth: true
                        text: serverHost
                        placeholderText: "Pi IP"
                        font.pointSize: ScreenTools.smallFontPointSize
                        onTextChanged: serverHost = text
                    }

                    QGCLabel {
                        text: ":"
                        font.pointSize: ScreenTools.smallFontPointSize
                    }

                    QGCTextField {
                        id: portInput
                        Layout.preferredWidth: ScreenTools.defaultFontPixelWidth * 5
                        text: serverPort.toString()
                        font.pointSize: ScreenTools.smallFontPointSize
                        validator: IntValidator { bottom: 1; top: 65535 }
                        onTextChanged: {
                            var p = parseInt(text)
                            if (p > 0 && p <= 65535) serverPort = p
                        }
                    }
                }

                // Connect button
                QGCButton {
                    Layout.fillWidth: true
                    text: connecting ? "Connecting..." : (connected ? "Reconnect" : "Connect")
                    enabled: !connecting
                    onClicked: doConnect()
                }

                // Error display
                QGCLabel {
                    visible: lastError !== ""
                    Layout.fillWidth: true
                    text: lastError
                    font.pointSize: ScreenTools.smallFontPointSize
                    color: qgcPal.colorOrange
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }

        // Main controls - only when connected
        ColumnLayout {
            Layout.fillWidth: true
            spacing: _margins * 0.5
            visible: connected

            // Status display - compact
            Rectangle {
                Layout.fillWidth: true
                height: statusRow.height + _margins * 0.6
                color: trackingState === "locked" ? Qt.rgba(0, 0.5, 0, 0.2) :
                       trackingState === "lost" ? Qt.rgba(0.5, 0, 0, 0.2) :
                       Qt.rgba(0.5, 0.5, 0, 0.1)
                radius: 4

                RowLayout {
                    id: statusRow
                    anchors.centerIn: parent
                    spacing: _margins

                    QGCLabel {
                        text: trackingState.charAt(0).toUpperCase() + trackingState.slice(1)
                        font.bold: true
                        font.pointSize: ScreenTools.defaultFontPointSize
                        color: trackingState === "locked" ? qgcPal.colorGreen :
                               trackingState === "lost" ? qgcPal.colorRed :
                               trackingState === "acquiring" ? qgcPal.colorOrange : qgcPal.text
                    }

                    QGCLabel {
                        text: lockedTargetId >= 0 ? "#" + lockedTargetId : "(" + targetCount + ")"
                        font.pointSize: ScreenTools.smallFontPointSize
                        color: qgcPal.text
                    }
                }
            }

            // Lock / Unlock
            RowLayout {
                Layout.fillWidth: true
                spacing: _margins * 0.5

                QGCButton {
                    text: "Lock"
                    Layout.fillWidth: true
                    enabled: lockedTargetId < 0 && targetCount > 0
                    onClicked: sendCommand("/api/tracking/lock", null)
                }

                QGCButton {
                    text: "Unlock"
                    Layout.fillWidth: true
                    enabled: lockedTargetId >= 0
                    onClicked: sendCommand("/api/tracking/unlock", null)
                }
            }

            // Enable / Disable control
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
                               (enableBtn.enabled ? qgcPal.colorGreen : qgcPal.windowShade)
                        radius: 4
                        opacity: enableBtn.enabled || controlEnabled ? 1.0 : 0.4
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

            // Emergency Stop - always visible and enabled when connected
            QGCButton {
                Layout.fillWidth: true
                text: "STOP"

                background: Rectangle {
                    color: parent.pressed ? Qt.darker("#cc0000", 1.2) : "#cc0000"
                    radius: 4
                }
                contentItem: Text {
                    text: "STOP"
                    color: "white"
                    font.bold: true
                    font.pointSize: ScreenTools.defaultFontPointSize
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                onClicked: sendCommand("/api/emergency-stop", null)
            }
        }

        // Disconnected hint
        QGCLabel {
            visible: !connected && !showConfig
            Layout.fillWidth: true
            text: "Tap dot to connect"
            font.pointSize: ScreenTools.smallFontPointSize
            color: qgcPal.colorGrey
            horizontalAlignment: Text.AlignHCenter
        }
    }
}
