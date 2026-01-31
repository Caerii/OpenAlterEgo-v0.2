// OpenAlterEgoReceiver.cs
// Minimal websocket client for Unity.
// Tested conceptually with System.Net.WebSockets (Unity 2021+ / .NET 4.x).
//
// Expected server message JSON:
//   {"token":"yes","confidence":1.0,"t":1730000000.123}

using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Net.WebSockets;
using UnityEngine;

[Serializable]
public class OpenAlterEgoMessage {
    public string token;
    public float confidence;
    public double t;
}

public class OpenAlterEgoReceiver : MonoBehaviour
{
    public string ServerUrl = "ws://127.0.0.1:8765";

    public event Action<OpenAlterEgoMessage> OnToken;

    ClientWebSocket _ws;
    CancellationTokenSource _cts;

    async void Start()
    {
        _cts = new CancellationTokenSource();
        _ws = new ClientWebSocket();

        try
        {
            await _ws.ConnectAsync(new Uri(ServerUrl), _cts.Token);
            Debug.Log($"[OpenAlterEgo] Connected: {ServerUrl}");
            _ = ReceiveLoop();
        }
        catch (Exception e)
        {
            Debug.LogError($"[OpenAlterEgo] Connect error: {e}");
        }
    }

    async Task ReceiveLoop()
    {
        var buffer = new byte[4096];

        while (_ws != null && _ws.State == WebSocketState.Open && !_cts.IsCancellationRequested)
        {
            WebSocketReceiveResult result = null;
            var sb = new StringBuilder();

            do
            {
                result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "closing", _cts.Token);
                    Debug.Log("[OpenAlterEgo] Server closed connection.");
                    return;
                }
                sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
            }
            while (!result.EndOfMessage);

            var json = sb.ToString();
            try
            {
                var msg = JsonUtility.FromJson<OpenAlterEgoMessage>(json);
                if (msg != null && !string.IsNullOrEmpty(msg.token))
                {
                    OnToken?.Invoke(msg);
                    Debug.Log($"[OpenAlterEgo] token={msg.token} conf={msg.confidence}");
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[OpenAlterEgo] JSON parse failed: {e} :: {json}");
            }
        }
    }

    async void OnDestroy()
    {
        try
        {
            _cts?.Cancel();
            if (_ws != null && _ws.State == WebSocketState.Open)
            {
                await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "destroy", CancellationToken.None);
            }
        }
        catch { }
        finally
        {
            _ws?.Dispose();
        }
    }
}
