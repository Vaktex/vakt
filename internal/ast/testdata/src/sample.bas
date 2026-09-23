Attribute VB_Name = "Module1"
Option Explicit

Public Sub RunIt(ByVal cmd As String)
    Shell cmd
End Sub

Public Function Add(a As Long, b As Long) As Long
    Add = a + b
End Function
