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

Rectangle {
    id: root
    width: mainColumn.width + (_margins * 2)
    height: mainColumn.height + (_margins * 2)
    color: qgcPal.window
    radius: _radius
    border.color: connected ? qgcPal.colorGreen : qgcPal.windowShade
    border.width: connected ? 2 : 1

    // Configuration
    property string serverHost: "192.168.2.11"
    property int serverPort: 5000
    property string serverUrl: "http://" + serverHost + ":" + serverPort

    // Layout
    property real _margins: ScreenTools.defaultFontPixelWidth
    property real _panelWidth: ScreenTools.defaultFontPixelWidth * 20
    property real _radius: ScreenTools.defaultFontPixelWidth * 0.5
    property real _buttonHeight: ScreenTools.defaultFontPixelHeight * 2.2

    // State
    property bool connected: false
    property bool connecting: false
    property string lastError: ""
    property string trackingState: "searching"
    property bool controlEnabled: false
    property int lockedTargetId: -1
    property int targetCount: 0
    property var targets: []
    property bool expanded: true

    QGCPalette { id: qgcPal }

    // Auto-connect and polling
    Timer {
        id: pollTimer
        interval: 400
        running: root.visible
        repeat: true
        triggeredOnStart: true
        onTriggered: fetchStatus()
    }

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
                        lockedTargetId = (r.locked_target_id !== null && r.locked_target_id !== undefined) ? r.locked_target_id : -1
                        targets = r.targets || []
                        targetCount = targets.length
                    } catch (e) {
                        handleError("Parse error")
                    }
                } else {
                    handleError(xhr.status === 0 ? "Connection failed" : "HTTP " + xhr.status)
                }
            }
        }
        xhr.onerror = function() { connecting = false; handleError("Network error") }
        xhr.ontimeout = function() { connecting = false; handleError("Timeout") }

        connecting = true
        try {
            xhr.open("GET", serverUrl + "/api/status")
            xhr.timeout = 1500
            xhr.send()
        } catch (e) {
            connecting = false
            handleError("Request failed")
        }
    }

    function handleError(msg) {
        if (connected) {
            lastError = msg
        }
        connected = false
    }

    function sendCommand(endpoint) {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                fetchStatus()
            }
        }
        xhr.open("POST", serverUrl + endpoint)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.timeout = 2000
        xhr.send("")
    }

    ColumnLayout {
        id: mainColumn
        anchors.centerIn: parent
        spacing: _margins * 0.6
        width: _panelWidth

        // Header row
        RowLayout {
            Layout.fillWidth: true
            spacing: _margins * 0.5

            // Status indicator
            Rectangle {
                width: ScreenTools.defaultFontPixelWidth * 0.8
                height: width
                radius: width / 2
                color: {
                    if (!connected) return qgcPal.colorRed
                    if (controlEnabled) return qgcPal.colorGreen
                    if (trackingState === "locked") return qgcPal.colorBlue
                    return qgcPal.colorOrange
                }

                SequentialAnimation on opacity {
                    running: connecting || (connected && trackingState === "acquiring")
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.3; duration: 250 }
                    NumberAnimation { to: 1.0; duration: 250 }
                }
            }

            QGCLabel {
                text: {
                    if (!connected) return "Tracking"
                    if (controlEnabled) return "ACTIVE"
                    return trackingState.charAt(0).toUpperCase() + trackingState.slice(1)
                }
                font.pointSize: ScreenTools.defaultFontPointSize
                font.bold: true
                color: {
                    if (!connected) return qgcPal.text
                    if (controlEnabled) return qgcPal.colorGreen
                    if (trackingState === "locked") return qgcPal.colorBlue
                    if (trackingState === "lost") return qgcPal.colorRed
                    return qgcPal.text
                }
                Layout.fillWidth: true
            }

            // Target info
            QGCLabel {
                visible: connected
                text: lockedTargetId >= 0 ? "#" + lockedTargetId : targetCount > 0 ? targetCount.toString() : "-"
                font.pointSize: ScreenTools.smallFontPointSize
                color: lockedTargetId >= 0 ? qgcPal.colorGreen : qgcPal.text
            }

            // Expand/collapse
            Rectangle {
                width: ScreenTools.defaultFontPixelWidth * 1.5
                height: width
                radius: 2
                color: expandMouse.containsMouse ? Qt.darker(qgcPal.window, 1.1) : "transparent"

                QGCLabel {
                    anchors.centerIn: parent
                    text: expanded ? "-" : "+"
                    font.pointSize: ScreenTools.defaultFontPointSize
                    font.bold: true
                }

                MouseArea {
                    id: expandMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: expanded = !expanded
                }
            }
        }

        // Expandable content
        ColumnLayout {
            Layout.fillWidth: true
            spacing: _margins * 0.5
            visible: expanded

            // Connection config (only when disconnected)
            Rectangle {
                Layout.fillWidth: true
                height: !connected ? configCol.height + _margins * 0.8 : 0
                color: Qt.darker(qgcPal.window, 1.08)
                radius: _radius
                clip: true
                visible: height > 0

                Behavior on height { NumberAnimation { duration: 150 } }

                ColumnLayout {
                    id: configCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: _margins * 0.5
                    spacing: _margins * 0.3

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 2

                        QGCTextField {
                            Layout.fillWidth: true
                            text: serverHost
                            placeholderText: "IP"
                            font.pointSize: ScreenTools.smallFontPointSize
                            onTextChanged: { serverHost = text; serverUrl = "http://" + serverHost + ":" + serverPort }
                        }

                        QGCLabel { text: ":"; font.pointSize: ScreenTools.smallFontPointSize }

                        QGCTextField {
                            Layout.preferredWidth: ScreenTools.defaultFontPixelWidth * 4.5
                            text: serverPort.toString()
                            font.pointSize: ScreenTools.smallFontPointSize
                            validator: IntValidator { bottom: 1; top: 65535 }
                            onTextChanged: {
                                var p = parseInt(text)
                                if (p > 0 && p <= 65535) {
                                    serverPort = p
                                    serverUrl = "http://" + serverHost + ":" + serverPort
                                }
                            }
                        }
                    }

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

            // Controls (only when connected)
            ColumnLayout {
                Layout.fillWidth: true
                spacing: _margins * 0.4
                visible: connected

                // Lock/Unlock row
                RowLayout {
                    Layout.fillWidth: true
                    spacing: _margins * 0.4

                    Rectangle {
                        Layout.fillWidth: true
                        height: _buttonHeight
                        radius: _radius
                        color: {
                            if (lockedTargetId >= 0) return Qt.darker(qgcPal.colorGreen, 1.3)
                            if (targetCount === 0) return qgcPal.windowShade
                            return lockMouse.pressed ? Qt.darker(qgcPal.colorBlue, 1.2) : qgcPal.colorBlue
                        }
                        opacity: (lockedTargetId < 0 && targetCount > 0) || lockedTargetId >= 0 ? 1.0 : 0.5

                        QGCLabel {
                            anchors.centerIn: parent
                            text: lockedTargetId >= 0 ? "LOCKED" : "Lock"
                            font.bold: true
                            font.pointSize: ScreenTools.defaultFontPointSize
                            color: "white"
                        }

                        MouseArea {
                            id: lockMouse
                            anchors.fill: parent
                            enabled: lockedTargetId < 0 && targetCount > 0
                            cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                            onClicked: sendCommand("/api/tracking/lock")
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: _buttonHeight
                        radius: _radius
                        color: unlockMouse.pressed ? Qt.darker(qgcPal.windowShade, 1.2) : qgcPal.windowShade
                        opacity: lockedTargetId >= 0 ? 1.0 : 0.4

                        QGCLabel {
                            anchors.centerIn: parent
                            text: "Unlock"
                            font.pointSize: ScreenTools.defaultFontPointSize
                            color: lockedTargetId >= 0 ? qgcPal.text : Qt.darker(qgcPal.text, 1.5)
                        }

                        MouseArea {
                            id: unlockMouse
                            anchors.fill: parent
                            enabled: lockedTargetId >= 0
                            cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                            onClicked: sendCommand("/api/tracking/unlock")
                        }
                    }
                }

                // Target selection (when multiple targets available)
                Repeater {
                    model: targets.length > 1 ? targets.slice(0, 4) : []

                    Rectangle {
                        Layout.fillWidth: true
                        height: ScreenTools.defaultFontPixelHeight * 1.8
                        radius: _radius
                        color: {
                            if (lockedTargetId === modelData.id) return Qt.darker(qgcPal.colorGreen, 1.3)
                            return targetSelectMouse.pressed ? Qt.darker(qgcPal.windowShade, 1.2) : qgcPal.windowShade
                        }

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: _margins * 0.3
                            spacing: _margins * 0.3

                            QGCLabel {
                                text: modelData["class"] || "Target"
                                font.pointSize: ScreenTools.smallFontPointSize
                                font.bold: lockedTargetId === modelData.id
                                color: lockedTargetId === modelData.id ? "white" : qgcPal.text
                                Layout.fillWidth: true
                            }

                            QGCLabel {
                                text: Math.round((modelData.confidence || 0) * 100) + "%"
                                font.pointSize: ScreenTools.smallFontPointSize
                                color: lockedTargetId === modelData.id ? "white" : qgcPal.colorGrey
                            }

                            QGCLabel {
                                text: "#" + modelData.id
                                font.pointSize: ScreenTools.smallFontPointSize
                                font.bold: true
                                color: lockedTargetId === modelData.id ? "white" : qgcPal.text
                            }
                        }

                        MouseArea {
                            id: targetSelectMouse
                            anchors.fill: parent
                            enabled: lockedTargetId !== modelData.id
                            cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                            onClicked: {
                                var xhr = new XMLHttpRequest()
                                xhr.onreadystatechange = function() {
                                    if (xhr.readyState === XMLHttpRequest.DONE) fetchStatus()
                                }
                                xhr.open("POST", serverUrl + "/api/tracking/lock")
                                xhr.setRequestHeader("Content-Type", "application/json")
                                xhr.timeout = 2000
                                xhr.send(JSON.stringify({"target_id": modelData.id}))
                            }
                        }
                    }
                }

                // Enable/Disable row
                RowLayout {
                    Layout.fillWidth: true
                    spacing: _margins * 0.4

                    Rectangle {
                        id: enableButton
                        Layout.fillWidth: true
                        height: _buttonHeight
                        radius: _radius
                        color: {
                            if (controlEnabled) return qgcPal.colorGreen
                            if (lockedTargetId < 0) return qgcPal.windowShade
                            return enableMouse.pressed ? Qt.darker(qgcPal.colorGreen, 1.2) : qgcPal.colorGreen
                        }
                        opacity: controlEnabled || lockedTargetId >= 0 ? 1.0 : 0.4

                        SequentialAnimation on opacity {
                            running: controlEnabled
                            loops: Animation.Infinite
                            NumberAnimation { to: 0.7; duration: 800 }
                            NumberAnimation { to: 1.0; duration: 800 }
                        }

                        QGCLabel {
                            anchors.centerIn: parent
                            text: controlEnabled ? "ACTIVE" : "Enable"
                            font.bold: true
                            font.pointSize: ScreenTools.defaultFontPointSize
                            color: "white"
                        }

                        MouseArea {
                            id: enableMouse
                            anchors.fill: parent
                            enabled: !controlEnabled && lockedTargetId >= 0
                            cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                            onClicked: sendCommand("/api/tracking/enable")
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: _buttonHeight
                        radius: _radius
                        color: disableMouse.pressed ? Qt.darker(qgcPal.windowShade, 1.2) : qgcPal.windowShade
                        opacity: controlEnabled ? 1.0 : 0.4

                        QGCLabel {
                            anchors.centerIn: parent
                            text: "Disable"
                            font.pointSize: ScreenTools.defaultFontPointSize
                            color: controlEnabled ? qgcPal.text : Qt.darker(qgcPal.text, 1.5)
                        }

                        MouseArea {
                            id: disableMouse
                            anchors.fill: parent
                            enabled: controlEnabled
                            cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                            onClicked: sendCommand("/api/tracking/disable")
                        }
                    }
                }

                // Emergency Stop
                Rectangle {
                    Layout.fillWidth: true
                    height: _buttonHeight
                    radius: _radius
                    color: stopMouse.pressed ? "#990000" : "#cc0000"

                    QGCLabel {
                        anchors.centerIn: parent
                        text: "STOP"
                        font.bold: true
                        font.pointSize: ScreenTools.defaultFontPointSize
                        color: "white"
                    }

                    MouseArea {
                        id: stopMouse
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: sendCommand("/api/emergency-stop")
                    }
                }
            }
        }
    }
}
