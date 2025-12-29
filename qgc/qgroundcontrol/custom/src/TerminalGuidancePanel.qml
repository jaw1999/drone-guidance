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
    width: Math.max(panelColumn.width + (_margins * 2), ScreenTools.defaultFontPixelWidth * 25)
    height: panelColumn.height + (_margins * 2)
    color: qgcPal.window
    radius: ScreenTools.defaultFontPixelWidth * 0.5
    border.color: qgcPal.windowShade
    border.width: 1

    // Server configuration - separate IP and port for clarity
    property string serverHost: "127.0.0.1"
    property int serverPort: 5000
    property string serverUrl: "http://" + serverHost + ":" + serverPort
    property real _margins: ScreenTools.defaultFontPixelWidth

    // Validation constants
    readonly property int minPort: 1
    readonly property int maxPort: 65535
    readonly property int maxTargets: 10  // Max targets to display
    readonly property int pollInterval: 500  // ms
    readonly property int requestTimeout: 2000  // ms

    // State from Terminal Guidance server
    property bool connected: false
    property string trackingState: "searching"
    property bool controlEnabled: false
    property int lockedTargetId: -1
    property var targets: []
    property string lastError: ""
    property bool requestInProgress: false

    // Show/hide settings panel
    property bool showSettings: false

    // Poll status when panel is visible
    Timer {
        id: statusTimer
        interval: pollInterval
        running: root.visible && !requestInProgress
        repeat: true
        onTriggered: fetchStatus()
    }

    Component.onCompleted: {
        // Validate initial values
        validateServerConfig()
        fetchStatus()
    }

    // Release focus when panel becomes hidden to prevent view switching issues
    onVisibleChanged: {
        if (!visible) {
            showSettings = false
            // Force release focus from any text fields
            root.forceActiveFocus()
        }
    }

    // Validate server configuration
    function validateServerConfig() {
        // Validate port
        if (serverPort < minPort || serverPort > maxPort) {
            serverPort = 5000
        }
        // Validate host - basic check for non-empty
        if (!serverHost || serverHost.trim().length === 0) {
            serverHost = "192.168.1.100"
        }
    }

    // Validate IP address format (basic validation)
    function isValidIPv4(ip) {
        if (!ip || typeof ip !== 'string') return false
        var parts = ip.trim().split('.')
        if (parts.length !== 4) return false
        for (var i = 0; i < 4; i++) {
            var num = parseInt(parts[i], 10)
            if (isNaN(num) || num < 0 || num > 255) return false
            if (parts[i] !== num.toString()) return false  // No leading zeros
        }
        return true
    }

    // Validate hostname (allows IP or hostname)
    function isValidHost(host) {
        if (!host || typeof host !== 'string') return false
        host = host.trim()
        if (host.length === 0 || host.length > 253) return false
        // Allow IP addresses
        if (isValidIPv4(host)) return true
        // Allow hostnames (basic check)
        var hostnameRegex = /^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$/
        return hostnameRegex.test(host)
    }

    function fetchStatus() {
        if (requestInProgress) return

        var xhr = new XMLHttpRequest()
        requestInProgress = true

        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                requestInProgress = false
                if (xhr.status === 200) {
                    try {
                        var response = JSON.parse(xhr.responseText)
                        connected = true
                        lastError = ""

                        // Validate and sanitize response data
                        trackingState = sanitizeTrackingState(response.tracking_state)
                        controlEnabled = response.control_enabled === true
                        lockedTargetId = sanitizeTargetId(response.locked_target_id)
                        targets = sanitizeTargets(response.targets)
                    } catch (e) {
                        connected = false
                        lastError = "Invalid response"
                        resetState()
                    }
                } else {
                    connected = false
                    if (xhr.status === 0) {
                        lastError = "Connection failed"
                    } else if (xhr.status >= 400 && xhr.status < 500) {
                        lastError = "Client error (" + xhr.status + ")"
                    } else if (xhr.status >= 500) {
                        lastError = "Server error (" + xhr.status + ")"
                    } else {
                        lastError = "HTTP " + xhr.status
                    }
                    resetState()
                }
            }
        }

        xhr.onerror = function() {
            requestInProgress = false
            connected = false
            lastError = "Network error"
            resetState()
        }

        xhr.ontimeout = function() {
            requestInProgress = false
            connected = false
            lastError = "Timeout"
            resetState()
        }

        try {
            xhr.open("GET", serverUrl + "/api/status")
            xhr.timeout = requestTimeout
            xhr.send()
        } catch (e) {
            requestInProgress = false
            connected = false
            lastError = "Request failed"
            resetState()
        }
    }

    // Reset state to safe defaults
    function resetState() {
        trackingState = "searching"
        controlEnabled = false
        lockedTargetId = -1
        targets = []
    }

    // Sanitize tracking state
    function sanitizeTrackingState(state) {
        var validStates = ["searching", "tracking", "locked", "lost"]
        if (typeof state === 'string' && validStates.indexOf(state.toLowerCase()) !== -1) {
            return state.toLowerCase()
        }
        return "searching"
    }

    // Sanitize target ID
    function sanitizeTargetId(id) {
        if (id === null || id === undefined) return -1
        var num = parseInt(id, 10)
        if (isNaN(num) || num < 0) return -1
        return num
    }

    // Sanitize targets array
    function sanitizeTargets(targetsArray) {
        if (!Array.isArray(targetsArray)) return []
        var sanitized = []
        var count = Math.min(targetsArray.length, maxTargets)
        for (var i = 0; i < count; i++) {
            var t = targetsArray[i]
            if (t && typeof t === 'object') {
                sanitized.push({
                    id: sanitizeTargetId(t.id),
                    class: (typeof t.class === 'string') ? t.class.substring(0, 32) : "unknown",
                    confidence: Math.max(0, Math.min(1, parseFloat(t.confidence) || 0))
                })
            }
        }
        return sanitized
    }

    function sendCommand(endpoint, body) {
        if (!connected) return

        var xhr = new XMLHttpRequest()
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                // Refresh status after command regardless of result
                Qt.callLater(fetchStatus)
            }
        }

        xhr.onerror = function() {
            lastError = "Command failed"
        }

        try {
            xhr.open("POST", serverUrl + endpoint)
            xhr.setRequestHeader("Content-Type", "application/json")
            xhr.timeout = requestTimeout
            xhr.send(body ? JSON.stringify(body) : "")
        } catch (e) {
            lastError = "Command error"
        }
    }

    ColumnLayout {
        id: panelColumn
        anchors.centerIn: parent
        spacing: _margins

        // Header with connection status and settings toggle
        RowLayout {
            Layout.fillWidth: true
            spacing: _margins

            QGCLabel {
                text: "Terminal Guidance"
                font.pointSize: ScreenTools.mediumFontPointSize
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            // Connection indicator with tooltip
            Rectangle {
                width: ScreenTools.defaultFontPixelWidth * 1.2
                height: width
                radius: width / 2
                color: connected ? qgcPal.colorGreen : qgcPal.colorRed

                ToolTip.visible: connIndicatorMouse.containsMouse
                ToolTip.delay: 500
                ToolTip.text: connected ? "Connected to " + serverHost + ":" + serverPort : (lastError || "Disconnected")

                MouseArea {
                    id: connIndicatorMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: fetchStatus()  // Click to retry connection
                }
            }

            // Settings gear button
            QGCColoredImage {
                width: ScreenTools.defaultFontPixelHeight * 1.2
                height: width
                source: "/qmlimages/Gears.svg"
                color: settingsMouseArea.containsMouse ? qgcPal.buttonHighlight : qgcPal.text
                fillMode: Image.PreserveAspectFit

                MouseArea {
                    id: settingsMouseArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: showSettings = !showSettings
                }
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
            QGCLabel {
                text: targets.length + (targets.length === 1 ? " object" : " objects")
            }
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
                enabled: connected && lockedTargetId < 0 && targets.length > 0
                onClicked: sendCommand("/api/tracking/lock", null)
            }

            QGCButton {
                text: "Unlock"
                Layout.fillWidth: true
                enabled: connected && lockedTargetId >= 0
                onClicked: sendCommand("/api/tracking/unlock", null)
            }
        }

        // Target list (show up to 4 targets for manual selection)
        Repeater {
            model: {
                if (!connected || targets.length === 0) return []
                return targets.slice(0, 4)
            }

            QGCButton {
                Layout.fillWidth: true
                text: {
                    var className = modelData.class || "Unknown"
                    var id = modelData.id >= 0 ? modelData.id : "?"
                    var conf = modelData.confidence ? " (" + Math.round(modelData.confidence * 100) + "%)" : ""
                    return "Lock: " + className + " #" + id + conf
                }
                enabled: connected && lockedTargetId !== modelData.id
                highlighted: lockedTargetId === modelData.id

                onClicked: {
                    if (modelData.id >= 0) {
                        sendCommand("/api/tracking/lock", {"target_id": modelData.id})
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
                enabled: connected && !controlEnabled && lockedTargetId >= 0

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

        // Emergency Stop - always prominent
        QGCButton {
            Layout.fillWidth: true
            Layout.preferredHeight: ScreenTools.defaultFontPixelHeight * 2.5
            text: "EMERGENCY STOP"
            enabled: connected

            background: Rectangle {
                color: parent.enabled ? (parent.pressed ? Qt.darker("#cc0000", 1.2) : "#cc0000") : qgcPal.windowShade
                radius: ScreenTools.defaultFontPixelWidth * 0.5
                border.color: parent.enabled ? Qt.darker("#cc0000", 1.3) : qgcPal.windowShade
                border.width: 1
            }

            contentItem: Text {
                text: parent.text
                color: parent.enabled ? "white" : qgcPal.colorGrey
                font.bold: true
                font.pointSize: ScreenTools.defaultFontPointSize
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }

            onClicked: sendCommand("/api/emergency-stop", null)
        }

        // Connection info footer - clickable to open settings
        QGCLabel {
            Layout.fillWidth: true
            text: connected ? serverHost + ":" + serverPort : (lastError || "Not connected")
            font.pointSize: ScreenTools.smallFontPointSize
            color: connected ? qgcPal.colorGrey : qgcPal.colorOrange
            horizontalAlignment: Text.AlignHCenter
            elide: Text.ElideMiddle

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: showSettings = !showSettings
            }
        }
    }

    // Server Settings Popup - positioned as overlay on top of the panel
    Rectangle {
        id: settingsPopup
        visible: showSettings
        width: parent.width - (_margins * 2)
        height: settingsColumn.height + (_margins * 2)
        x: _margins
        y: panelColumn.y + ScreenTools.defaultFontPixelHeight * 3  // Below header
        z: 100  // On top of everything
        color: qgcPal.window
        radius: ScreenTools.defaultFontPixelWidth * 0.5
        border.color: qgcPal.buttonHighlight
        border.width: 2

        // Shadow effect
        Rectangle {
            anchors.fill: parent
            anchors.margins: -2
            z: -1
            color: "transparent"
            border.color: Qt.rgba(0, 0, 0, 0.3)
            border.width: 4
            radius: parent.radius + 2
        }

        ColumnLayout {
            id: settingsColumn
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: _margins
            spacing: _margins * 0.75

            // Header row with title and close button
            RowLayout {
                Layout.fillWidth: true

                QGCLabel {
                    text: "Server Connection"
                    font.pointSize: ScreenTools.defaultFontPointSize
                    font.bold: true
                }

                Item { Layout.fillWidth: true }

                // Close button
                Rectangle {
                    width: ScreenTools.defaultFontPixelHeight * 1.5
                    height: width
                    radius: width / 2
                    color: closeMouseArea.containsMouse ? qgcPal.buttonHighlight : "transparent"

                    QGCLabel {
                        anchors.centerIn: parent
                        text: "✕"
                        font.pointSize: ScreenTools.smallFontPointSize
                        font.bold: true
                    }

                    MouseArea {
                        id: closeMouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: showSettings = false
                    }
                }
            }

            // Separator
            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: qgcPal.windowShade
            }

            // IP Address row
            RowLayout {
                Layout.fillWidth: true
                spacing: _margins / 2

                QGCLabel {
                    text: "IP Address:"
                    font.pointSize: ScreenTools.smallFontPointSize
                    Layout.preferredWidth: ScreenTools.defaultFontPixelWidth * 10
                }

                QGCTextField {
                    id: hostField
                    Layout.fillWidth: true
                    text: serverHost
                    font.pointSize: ScreenTools.smallFontPointSize
                    placeholderText: "192.168.1.100"
                    maximumLength: 253
                    validationError: false  // Never block view switching

                    property bool isValid: isValidHost(text)

                    background: Rectangle {
                        color: hostField.isValid ? qgcPal.window : Qt.rgba(1, 0.6, 0.6, 0.3)
                        border.color: hostField.activeFocus ? qgcPal.buttonHighlight :
                                     (hostField.isValid ? qgcPal.windowShade : qgcPal.colorRed)
                        border.width: 1
                        radius: 3
                    }

                    onEditingFinished: {
                        var trimmed = text.trim()
                        if (isValidHost(trimmed)) {
                            root.serverHost = trimmed
                            fetchStatus()
                        } else {
                            text = serverHost  // Revert to last valid
                        }
                    }

                    Keys.onReturnPressed: focus = false
                    Keys.onEscapePressed: { text = serverHost; focus = false }
                }
            }

            // Port row
            RowLayout {
                Layout.fillWidth: true
                spacing: _margins / 2

                QGCLabel {
                    text: "Port:"
                    font.pointSize: ScreenTools.smallFontPointSize
                    Layout.preferredWidth: ScreenTools.defaultFontPixelWidth * 10
                }

                QGCTextField {
                    id: portField
                    Layout.fillWidth: true
                    text: serverPort.toString()
                    font.pointSize: ScreenTools.smallFontPointSize
                    placeholderText: "5000"
                    inputMethodHints: Qt.ImhDigitsOnly
                    maximumLength: 5
                    validator: IntValidator { bottom: minPort; top: maxPort }
                    validationError: false  // Never block view switching

                    property bool isValid: {
                        var num = parseInt(text, 10)
                        return !isNaN(num) && num >= minPort && num <= maxPort
                    }

                    background: Rectangle {
                        color: portField.isValid ? qgcPal.window : Qt.rgba(1, 0.6, 0.6, 0.3)
                        border.color: portField.activeFocus ? qgcPal.buttonHighlight :
                                     (portField.isValid ? qgcPal.windowShade : qgcPal.colorRed)
                        border.width: 1
                        radius: 3
                    }

                    onEditingFinished: {
                        var port = parseInt(text, 10)
                        if (!isNaN(port) && port >= minPort && port <= maxPort) {
                            root.serverPort = port
                            fetchStatus()
                        } else {
                            text = serverPort.toString()  // Revert to last valid
                        }
                    }

                    Keys.onReturnPressed: focus = false
                    Keys.onEscapePressed: { text = serverPort.toString(); focus = false }
                }
            }

            // Connection status and test button
            RowLayout {
                Layout.fillWidth: true
                spacing: _margins / 2

                // Status indicator
                Rectangle {
                    width: ScreenTools.defaultFontPixelWidth * 1.2
                    height: width
                    radius: width / 2
                    color: connected ? qgcPal.colorGreen : qgcPal.colorRed
                }

                QGCLabel {
                    text: connected ? "Connected" : (lastError || "Not connected")
                    font.pointSize: ScreenTools.smallFontPointSize
                    color: connected ? qgcPal.colorGreen : qgcPal.colorOrange
                    Layout.fillWidth: true
                }

                QGCButton {
                    text: requestInProgress ? "..." : "Connect"
                    enabled: !requestInProgress && hostField.isValid && portField.isValid
                    Layout.preferredWidth: ScreenTools.defaultFontPixelWidth * 10

                    onClicked: fetchStatus()
                }
            }
        }
    }

}
