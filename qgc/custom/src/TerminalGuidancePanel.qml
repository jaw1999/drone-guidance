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

    property string serverUrl: "http://192.168.1.100:5000"  // Default, configure in settings
    property real _margins: ScreenTools.defaultFontPixelWidth

    // State from Terminal Guidance server
    property bool connected: false
    property string trackingState: "searching"
    property bool controlEnabled: false
    property int lockedTargetId: -1
    property var targets: []

    QGCPalette { id: qgcPal }

    // Poll status every 500ms when panel is visible
    Timer {
        id: statusTimer
        interval: 500
        running: root.visible
        repeat: true
        onTriggered: fetchStatus()
    }

    Component.onCompleted: fetchStatus()

    function fetchStatus() {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    try {
                        var response = JSON.parse(xhr.responseText)
                        connected = true
                        trackingState = response.tracking_state || "searching"
                        controlEnabled = response.control_enabled || false
                        lockedTargetId = response.locked_target_id !== null ? response.locked_target_id : -1
                        targets = response.targets || []
                    } catch (e) {
                        connected = false
                    }
                } else {
                    connected = false
                }
            }
        }
        xhr.open("GET", serverUrl + "/api/status")
        xhr.timeout = 1000
        xhr.send()
    }

    function sendCommand(endpoint, body) {
        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                fetchStatus()  // Refresh status after command
            }
        }
        xhr.open("POST", serverUrl + endpoint)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.timeout = 2000
        xhr.send(body ? JSON.stringify(body) : "")
    }

    ColumnLayout {
        id: panelColumn
        anchors.centerIn: parent
        spacing: _margins

        // Header with connection status
        RowLayout {
            Layout.fillWidth: true
            spacing: _margins

            QGCLabel {
                text: "Terminal Guidance"
                font.pointSize: ScreenTools.mediumFontPointSize
                font.bold: true
            }

            Rectangle {
                width: ScreenTools.defaultFontPixelWidth
                height: width
                radius: width / 2
                color: connected ? qgcPal.colorGreen : qgcPal.colorRed
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: qgcPal.windowShade
        }

        // Status display
        GridLayout {
            columns: 2
            columnSpacing: _margins
            rowSpacing: _margins / 2

            QGCLabel { text: "State:" }
            QGCLabel {
                text: trackingState.charAt(0).toUpperCase() + trackingState.slice(1)
                color: {
                    switch(trackingState) {
                        case "locked": return qgcPal.colorGreen
                        case "tracking": return qgcPal.colorOrange
                        case "lost": return qgcPal.colorRed
                        default: return qgcPal.text
                    }
                }
                font.bold: true
            }

            QGCLabel { text: "Control:" }
            QGCLabel {
                text: controlEnabled ? "ENABLED" : "Disabled"
                color: controlEnabled ? qgcPal.colorGreen : qgcPal.text
                font.bold: controlEnabled
            }

            QGCLabel { text: "Target:" }
            QGCLabel {
                text: lockedTargetId >= 0 ? "ID " + lockedTargetId : "None"
                color: lockedTargetId >= 0 ? qgcPal.colorGreen : qgcPal.text
            }

            QGCLabel { text: "Detected:" }
            QGCLabel { text: targets.length + " objects" }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: qgcPal.windowShade
        }

        // Target Lock Controls
        QGCLabel {
            text: "Target Lock"
            font.pointSize: ScreenTools.smallFontPointSize
            color: qgcPal.colorGrey
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: _margins / 2

            QGCButton {
                text: "Auto"
                Layout.fillWidth: true
                enabled: connected && lockedTargetId < 0
                onClicked: sendCommand("/api/tracking/lock", null)
            }

            QGCButton {
                text: "Unlock"
                Layout.fillWidth: true
                enabled: connected && lockedTargetId >= 0
                onClicked: sendCommand("/api/tracking/unlock", null)
            }
        }

        // Target list (if multiple targets)
        Repeater {
            model: targets.length > 0 && targets.length <= 4 ? targets : []

            QGCButton {
                Layout.fillWidth: true
                text: "Lock: " + modelData.class + " (ID " + modelData.id + ")"
                enabled: connected && lockedTargetId !== modelData.id
                highlighted: lockedTargetId === modelData.id

                onClicked: sendCommand("/api/tracking/lock", {"target_id": modelData.id})
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: qgcPal.windowShade
        }

        // Tracking Control
        QGCLabel {
            text: "Flight Control"
            font.pointSize: ScreenTools.smallFontPointSize
            color: qgcPal.colorGrey
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: _margins / 2

            QGCButton {
                text: "Enable"
                Layout.fillWidth: true
                enabled: connected && !controlEnabled
                backRadius: ScreenTools.defaultFontPixelWidth * 0.5

                onClicked: sendCommand("/api/tracking/enable", null)
            }

            QGCButton {
                text: "Disable"
                Layout.fillWidth: true
                enabled: connected && controlEnabled

                onClicked: sendCommand("/api/tracking/disable", null)
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: qgcPal.windowShade
        }

        // Emergency Stop
        QGCButton {
            Layout.fillWidth: true
            text: "EMERGENCY STOP"
            enabled: connected
            backRadius: ScreenTools.defaultFontPixelWidth * 0.5

            background: Rectangle {
                color: parent.pressed ? Qt.darker(qgcPal.colorRed, 1.2) : qgcPal.colorRed
                radius: ScreenTools.defaultFontPixelWidth * 0.5
            }

            contentItem: Text {
                text: parent.text
                color: "white"
                font.bold: true
                font.pointSize: ScreenTools.defaultFontPointSize
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }

            onClicked: sendCommand("/api/emergency-stop", null)
        }

        // Server URL config
        RowLayout {
            Layout.fillWidth: true
            spacing: _margins / 2

            QGCLabel {
                text: "Server:"
                font.pointSize: ScreenTools.smallFontPointSize
            }

            QGCTextField {
                id: serverUrlField
                Layout.fillWidth: true
                text: serverUrl
                font.pointSize: ScreenTools.smallFontPointSize

                onEditingFinished: {
                    root.serverUrl = text
                    fetchStatus()
                }
            }
        }
    }
}
