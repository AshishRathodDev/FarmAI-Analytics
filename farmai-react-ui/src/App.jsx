import React, { useState, useCallback } from 'react';
import { Camera, Home, MessageSquare, BarChart3, User, Settings, Sun, Moon, Bell, Search, Upload, Loader2, CheckCircle, AlertCircle, TrendingUp, Users, Zap, Target, ChevronRight, Calendar, MapPin, Leaf, X, Menu, Download, RefreshCw, Copy, Volume2, Send, Mic, Paperclip, Filter, ArrowUp, Eye, LogOut, Edit, Save } from 'lucide-react';
import { chatWithAI, predictDisease } from './api-service';


const mockDiseases = [
  { id: 1, name: "Tomato Late Blight", count: 234, severity: "high", color: "#EF4444" },
  { id: 2, name: "Potato Early Blight", count: 189, severity: "medium", color: "#F59E0B" },
  { id: 3, name: "Tomato Bacterial Spot", count: 156, severity: "medium", color: "#F59E0B" },
  { id: 4, name: "Pepper Bacterial Spot", count: 123, severity: "low", color: "#10B981" },
  { id: 5, name: "Tomato Septoria", count: 98, severity: "medium", color: "#F59E0B" }
];

const mockScans = [
  { id: 1, date: "2 hours ago", farmer: "Rajesh K.", disease: "Tomato Late Blight", confidence: 94.2 },
  { id: 2, date: "5 hours ago", farmer: "Priya S.", disease: "Healthy", confidence: 89.1 },
  { id: 3, date: "1 day ago", farmer: "Amit P.", disease: "Potato Early Blight", confidence: 91.5 }
];

const mockChartData = [
  { date: "Nov 1", scans: 45 },
  { date: "Nov 5", scans: 52 },
  { date: "Nov 10", scans: 68 },
  { date: "Nov 15", scans: 81 },
  { date: "Nov 18", scans: 95 }
];

export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [theme, setTheme] = useState('light');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [chatMessages, setChatMessages] = useState([
    { id: 1, type: 'bot', text: "Hello! I'm your FarmAI assistant. How can I help you today?", time: "10:23 AM" }
  ]);
  const [chatInput, setChatInput] = useState('');

  const toggleTheme = () => setTheme(theme === 'light' ? 'dark' : 'light');

  const handleImageUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      // Show preview
      const reader = new FileReader();
      reader.onload = (e) => {
      setUploadedImage(e.target.result);
      };
      reader.readAsDataURL(file);

      // Call Flask backend API
      setAnalyzing(true);
      const result = await predictDisease(file);
      setAnalyzing(false);

      if (result.status === "success") {
        setAnalysisResult({
          disease: result.disease,
          confidence: result.confidence_percentage,
          severity: result.confidence_percentage > 90 ? "high" : result.confidence_percentage > 70 ? "medium" : "low",
          treatment: result.recommendation
        });
      } else {
        alert(result.message || "Prediction failed. Please try again.");
        setAnalysisResult(null);
      }
    }
  }, []);


  const handleChatImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setChatMessages([...chatMessages, {
          id: chatMessages.length + 1,
          type: 'user',
          text: `📷 Uploaded: ${file.name}`,
          time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          image: event.target.result
        }]);
        setTimeout(() => {
          setChatMessages(prev => [...prev, {
            id: prev.length + 1,
            type: 'bot',
            text: "I can see your image! For detailed disease analysis, please use the Scanner page. I can provide general farming advice here.",
            time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
          }]);
        }, 1000);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSendMessage = async () => {
  if (chatInput.trim()) {
    const userMsg = {
      id: chatMessages.length + 1,
      type: 'user',
      text: chatInput,
      time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    };

    setChatMessages([...chatMessages, userMsg]);
    const userQuestion = chatInput;
    setChatInput('');

    // Fetch from Flask backend
    const aiResponse = await chatWithAI(userQuestion);

    setChatMessages(prev => [...prev, {
      id: prev.length + 1,
      type: 'bot',
      text: aiResponse,
      time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    }]);
  }
};


  const bgClass = theme === 'dark' ? 'bg-slate-900' : 'bg-gray-50';
  const textClass = theme === 'dark' ? 'text-gray-100' : 'text-slate-900';
  const cardBg = theme === 'dark' ? 'bg-slate-800' : 'bg-white';
  const borderClass = theme === 'dark' ? 'border-slate-700' : 'border-gray-200';

  const navItems = [
    { id: 'dashboard', icon: Home, label: 'Home' },
    { id: 'chat', icon: MessageSquare, label: 'AI Chat' },
    { id: 'scan', icon: Camera, label: 'Scan' },
    { id: 'analytics', icon: BarChart3, label: 'Analytics' },
    { id: 'profile', icon: User, label: 'Profile' },
    { id: 'settings', icon: Settings, label: 'Settings' }
  ];

  return (
    <div className={`min-h-screen ${bgClass} ${textClass} transition-colors duration-300`}>
      {/* Top Bar */}
      <header className={`fixed top-0 left-0 right-0 h-16 ${cardBg} border-b ${borderClass} z-50 shadow-sm`}>
        <div className="h-full px-4 flex items-center justify-between max-w-[1920px] mx-auto">
          <div className="flex items-center gap-4">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="lg:hidden p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
              <Menu className="w-6 h-6" />
            </button>
            <div className="flex items-center gap-2">
              <Leaf className="w-8 h-8 text-emerald-500" />
              <span className="text-xl font-bold bg-gradient-to-r from-emerald-500 to-emerald-600 bg-clip-text text-transparent">FarmAI</span>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-2 flex-1 max-w-md mx-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input type="text" placeholder="Search..." className={`w-full pl-10 pr-4 py-2 rounded-lg ${cardBg} border ${borderClass} focus:outline-none focus:ring-2 focus:ring-emerald-500`} />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button className="relative p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            <button onClick={toggleTheme} className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors">
              {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </button>
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center text-white font-semibold text-sm cursor-pointer">A</div>
          </div>
        </div>
      </header>

      {/* Desktop Sidebar */}
      <aside className={`hidden lg:flex fixed left-0 top-16 bottom-0 w-60 ${cardBg} border-r ${borderClass} flex-col shadow-sm`}>
        <nav className="flex-1 p-4">
          {navItems.map(item => (
            <button key={item.id} onClick={() => setCurrentPage(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all ${
                currentPage === item.id 
                  ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/30' 
                  : `hover:bg-gray-100 dark:hover:bg-slate-700 ${textClass}`
              }`}>
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className={`p-4 border-t ${borderClass}`}>
          <div className="text-xs text-gray-500">v1.0.0 • EfficientNetB0</div>
        </div>
      </aside>

      {/* Mobile Sidebar */}
      {sidebarOpen && (
        <div className="lg:hidden fixed inset-0 z-40 bg-black/50" onClick={() => setSidebarOpen(false)}>
          <aside className={`fixed left-0 top-0 bottom-0 w-64 ${cardBg} p-4`} onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6 pt-4">
              <div className="flex items-center gap-2">
                <Leaf className="w-6 h-6 text-emerald-500" />
                <span className="font-bold text-lg">FarmAI</span>
              </div>
              <button onClick={() => setSidebarOpen(false)}><X className="w-6 h-6" /></button>
            </div>
            <nav>
              {navItems.map(item => (
                <button key={item.id} onClick={() => { setCurrentPage(item.id); setSidebarOpen(false); }}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 ${
                    currentPage === item.id ? 'bg-emerald-500 text-white' : 'hover:bg-gray-100 dark:hover:bg-slate-700'
                  }`}>
                  <item.icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </aside>
        </div>
      )}

      {/* Mobile Bottom Nav */}
      <nav className={`lg:hidden fixed bottom-0 left-0 right-0 ${cardBg} border-t ${borderClass} z-50 shadow-lg`}>
        <div className="flex items-center justify-around px-2 py-2">
          {navItems.slice(0, 5).map(item => (
            <button key={item.id} onClick={() => setCurrentPage(item.id)}
              className={`flex flex-col items-center gap-1 px-3 py-2 rounded-lg min-w-[60px] ${
                currentPage === item.id ? 'text-emerald-500' : 'text-gray-500'
              }`}>
              <item.icon className="w-5 h-5" />
              <span className="text-xs font-medium">{item.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main className="lg:ml-60 mt-16 p-4 md:p-6 lg:p-8 pb-20 lg:pb-8 max-w-[1920px] mx-auto">
        
        {/* DASHBOARD PAGE */}
        {currentPage === 'dashboard' && (
          <div className="space-y-6">
            <div className="mb-8">
              <h1 className="text-3xl md:text-4xl font-bold mb-2 bg-gradient-to-r from-emerald-600 to-emerald-500 bg-clip-text text-transparent">
                Welcome back 👋
              </h1>
              <p className="text-gray-500 dark:text-gray-400">Here's what's happening with your crops today</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
              {[
                { icon: BarChart3, label: "Total Scans", value: "12,847", change: "+12%" },
                { icon: Users, label: "Farmers", value: "1,234", change: "+8%" },
                { icon: Zap, label: "Avg Speed", value: "80ms", change: "-5ms" },
                { icon: Target, label: "Accuracy", value: "95.3%", change: "+0.2%" }
              ].map((stat, i) => (
                <div key={i} className={`${cardBg} p-6 rounded-xl border ${borderClass} hover:shadow-xl hover:scale-105 transition-all duration-300 cursor-pointer`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-3 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 shadow-lg">
                      <stat.icon className="w-6 h-6 text-white" />
                    </div>
                    <span className={`text-sm font-semibold flex items-center gap-1 ${stat.change.includes('+') ? 'text-green-600' : 'text-red-600'}`}>
                      <ArrowUp className="w-4 h-4" /> {stat.change}
                    </span>
                  </div>
                  <div className="text-3xl font-bold mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
              {[
                { icon: Camera, title: "Scan New Crop", desc: "Upload leaf image for instant analysis", page: "scan" },
                { icon: MessageSquare, title: "Ask AI Assistant", desc: "Get expert farming advice 24/7", page: "chat" },
                { icon: BarChart3, title: "View Analytics", desc: "Track disease trends and patterns", page: "analytics" }
              ].map((action, i) => (
                <button key={i} onClick={() => setCurrentPage(action.page)}
                  className={`${cardBg} p-6 rounded-xl border ${borderClass} text-left hover:shadow-2xl transition-all duration-300 group hover:-translate-y-1`}>
                  <div className="inline-flex p-3 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 text-white mb-4 group-hover:scale-110 transition-transform shadow-lg">
                    <action.icon className="w-6 h-6" />
                  </div>
                  <h3 className="font-semibold text-lg mb-2">{action.title}</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">{action.desc}</p>
                  <div className="flex items-center text-emerald-600 font-medium text-sm group-hover:gap-2 transition-all">
                    Get Started <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                  </div>
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-sm`}>
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <div className="w-1 h-6 bg-emerald-500 rounded"></div>
                  Recent Activity
                </h3>
                <div className="space-y-4">
                  {mockScans.map(scan => (
                    <div key={scan.id} className="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center flex-shrink-0 shadow">
                        <Camera className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate">{scan.disease}</div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">{scan.farmer} • {scan.date}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold text-emerald-600">{scan.confidence}%</div>
                        <button className="text-sm text-gray-500 hover:text-emerald-600 transition-colors">View</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-sm`}>
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <div className="w-1 h-6 bg-emerald-500 rounded"></div>
                  Scan Volume Trend
                </h3>
                <div className="h-64 flex items-end justify-between gap-2">
                  {mockChartData.map((data, i) => (
                    <div key={i} className="flex-1 flex flex-col items-center gap-2 group">
                      <div className="w-full bg-gradient-to-t from-emerald-500 to-emerald-400 rounded-t-lg hover:from-emerald-600 hover:to-emerald-500 cursor-pointer transition-all shadow-lg group-hover:shadow-emerald-500/50" 
                           style={{ height: `${(data.scans / 100) * 100}%` }} 
                           title={`${data.date}: ${data.scans} scans`}>
                      </div>
                      <span className="text-xs text-gray-500 dark:text-gray-400">{data.date.split(' ')[1]}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-sm`}>
              <h3 className="font-semibold text-lg mb-6 flex items-center gap-2">
                <div className="w-1 h-6 bg-emerald-500 rounded"></div>
                Top Diseases This Week
              </h3>
              <div className="space-y-4">
                {mockDiseases.map((disease, i) => (
                  <div key={disease.id} className="flex items-center gap-4">
                    <span className="text-2xl font-bold text-gray-300 dark:text-gray-600 w-8">{i + 1}</span>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{disease.name}</span>
                        <span className="text-sm font-semibold text-gray-600 dark:text-gray-400">{disease.count} cases</span>
                      </div>
                      <div className="h-2 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-500" 
                             style={{ width: `${(disease.count / 234) * 100}%`, backgroundColor: disease.color }} />
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      disease.severity === 'high' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      disease.severity === 'medium' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400' : 
                      'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    }`}>
                      {disease.severity.toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* SCANNER PAGE */}
        {currentPage === 'scan' && (
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="mb-8">
              <h1 className="text-3xl md:text-4xl font-bold mb-2 bg-gradient-to-r from-emerald-600 to-emerald-500 bg-clip-text text-transparent">
                Disease Scanner
              </h1>
              <p className="text-gray-500 dark:text-gray-400">Upload a clear, well-lit image of the affected leaf for instant AI diagnosis</p>
            </div>

            {!uploadedImage ? (
              <div className={`${cardBg} p-8 md:p-12 rounded-xl border-2 border-dashed ${borderClass} text-center hover:border-emerald-500 transition-all`}>
                <label className="cursor-pointer block">
                  <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
                  <div className="py-12">
                    <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center">
                      <Upload className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-2xl font-semibold mb-2">Drag & Drop Image</h3>
                    <p className="text-gray-500 dark:text-gray-400 mb-6">or click to browse from your device</p>
                    <div className="inline-flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-lg hover:from-emerald-600 hover:to-emerald-700 font-medium shadow-lg hover:shadow-xl transition-all">
                      <Upload className="w-5 h-5" /> Browse Files
                    </div>
                    <p className="text-sm text-gray-400 mt-6">Supported: JPG, PNG, WEBP • Max size: 10MB • Best: Clear, natural lighting</p>
                  </div>
                </label>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-lg`}>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-lg">Uploaded Image</h3>
                    <button onClick={() => { setUploadedImage(null); setAnalysisResult(null); setAnalyzing(false); }}
                      className="px-4 py-2 text-sm border border-red-300 text-red-600 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors">
                      Reset
                    </button>
                  </div>
                  <div className="relative rounded-lg overflow-hidden mb-4 shadow-md">
                    <img src={uploadedImage} alt="Uploaded crop" className="w-full h-auto" />
                  </div>
                  <div className="flex gap-2">
                    <button className={`flex-1 px-4 py-2 border ${borderClass} rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors text-sm font-medium`}>
                      <RefreshCw className="w-4 h-4 inline mr-2" /> Rotate
                    </button>
                    <button className={`px-4 py-2 border ${borderClass} rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors`}>
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-lg`}>
                  <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-emerald-500" />
                    Analysis Results
                  </h3>
                  {analyzing ? (
                    <div className="text-center py-16">
                      <Loader2 className="w-16 h-16 animate-spin mx-auto mb-6 text-emerald-500" />
                      <p className="text-lg font-medium mb-2">Analyzing image...</p>
                      <p className="text-sm text-gray-500">Running AI model (EfficientNetB0)</p>
                      <div className="mt-6 max-w-xs mx-auto">
                        <div className="h-2 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-emerald-400 to-emerald-600 rounded-full animate-pulse" style={{width: '70%'}}></div>
                        </div>
                      </div>
                    </div>
                  ) : analysisResult && (
                    <div className="space-y-6">
                      <div className="text-center py-8 bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-900/20 dark:to-emerald-800/20 rounded-xl">
                        <div className="inline-flex items-center justify-center w-28 h-28 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 text-white text-3xl font-bold mb-4 shadow-lg">
                          {analysisResult.confidence}%
                        </div>
                        <div className={`inline-block px-4 py-2 rounded-full text-sm font-semibold mb-3 ${
                          analysisResult.severity === 'high' ? 'bg-red-500 text-white' :
                          analysisResult.severity === 'medium' ? 'bg-amber-500 text-white' : 'bg-green-500 text-white'
                        } shadow-lg`}>
                          SEVERITY: {analysisResult.severity.toUpperCase()}
                        </div>
                        <h4 className="text-2xl font-bold mt-2">{analysisResult.disease}</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">High confidence detection</p>
                      </div>
                      
                      <div className={`border-t ${borderClass} pt-6`}>
                        <h4 className="font-semibold mb-3 flex items-center gap-2 text-lg">
                          <CheckCircle className="w-5 h-5 text-emerald-500" /> Recommended Treatment
                        </h4>
                        <p className="text-gray-600 dark:text-gray-300 leading-relaxed bg-gray-50 dark:bg-slate-700/50 p-4 rounded-lg">
                          {analysisResult.treatment}
                        </p>
                      </div>
                      
                      <div className="flex gap-3 pt-4">
                        <button className="flex-1 px-6 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-lg hover:from-emerald-600 hover:to-emerald-700 font-medium shadow-lg hover:shadow-xl transition-all">
                          <Save className="w-4 h-4 inline mr-2" /> Save Report
                        </button>
                        <button className={`px-6 py-3 border ${borderClass} rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors font-medium`}>
                          <Download className="w-4 h-4 inline mr-2" /> Export
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* CHAT PAGE */}
        {currentPage === 'chat' && (
          <div className="max-w-5xl mx-auto">
            <div className={`${cardBg} rounded-xl border ${borderClass} flex flex-col shadow-lg`} style={{height: 'calc(100vh - 12rem)'}}>
              <div className={`p-4 border-b ${borderClass} flex items-center justify-between bg-gradient-to-r from-emerald-500/10 to-emerald-600/10`}>
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center shadow-lg">
                    <MessageSquare className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold">FarmAI Assistant</h3>
                    <span className="text-sm text-green-600 dark:text-green-400 flex items-center gap-1">
                      <span className="w-2 h-2 bg-green-600 rounded-full animate-pulse"></span> Online
                    </span>
                  </div>
                </div>
                <button className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors">
                  <Volume2 className="w-5 h-5" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {chatMessages.map(msg => (
                  <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
                    <div className={`max-w-[80%] ${
                      msg.type === 'bot' 
                        ? 'bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-900/20 dark:to-emerald-800/20 rounded-2xl rounded-tl-none' 
                        : 'bg-gray-100 dark:bg-slate-700 rounded-2xl rounded-tr-none'
                    } p-4 shadow-md`}>
                      {msg.image && (
                        <img src={msg.image} alt="Uploaded" className="w-full rounded-lg mb-3 max-h-64 object-cover" />
                      )}
                      <p className="mb-2 leading-relaxed">{msg.text}</p>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500 dark:text-gray-400">{msg.time}</span>
                        {msg.type === 'bot' && (
                          <div className="flex gap-2">
                            <button className="text-xs text-gray-500 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">
                              <Copy className="w-3 h-3 inline mr-1" /> Copy
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className={`p-4 border-t ${borderClass} bg-gray-50 dark:bg-slate-800/50`}>
                <div className="flex flex-wrap gap-2 mb-3">
                  {["Common crop diseases", "Fertilizer recommendations", "Pest control methods"].map(s => (
                    <button key={s} onClick={() => setChatInput(s)}
                      className="px-4 py-2 text-sm bg-white dark:bg-slate-700 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 rounded-full hover:bg-emerald-50 dark:hover:bg-emerald-900/20 transition-colors shadow-sm">
                      {s}
                    </button>
                  ))}
                </div>
                <div className="flex gap-2">
                  <label className="p-3 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors cursor-pointer">
                    <input type="file" accept="image/*" className="hidden" onChange={handleChatImageUpload} />
                    <Paperclip className="w-5 h-5" />
                  </label>
                  <input type="text" value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Ask me anything about farming..." 
                    className={`flex-1 px-4 py-3 ${cardBg} border ${borderClass} rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 shadow-sm`} />
                  <button className="p-3 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors">
                    <Mic className="w-5 h-5" />
                  </button>
                  <button onClick={handleSendMessage} 
                    className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-lg hover:from-emerald-600 hover:to-emerald-700 shadow-lg hover:shadow-xl transition-all font-medium">
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ANALYTICS PAGE */}
        {currentPage === 'analytics' && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
              <div>
                <h1 className="text-3xl md:text-4xl font-bold mb-2 bg-gradient-to-r from-emerald-600 to-emerald-500 bg-clip-text text-transparent">
                  Analytics Dashboard
                </h1>
                <p className="text-gray-500 dark:text-gray-400">Comprehensive insights into disease detection patterns</p>
              </div>
              <button className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-lg hover:from-emerald-600 hover:to-emerald-700 flex items-center gap-2 font-medium shadow-lg hover:shadow-xl transition-all">
                <Download className="w-4 h-4" /> Export Report
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
              {[
                { label: "Total Scans", value: "12,847", icon: BarChart3 },
                { label: "Unique Farmers", value: "1,234", icon: Users },
                { label: "Avg Confidence", value: "92.4%", icon: Target },
                { label: "Top Disease", value: "T. Late Blight", icon: AlertCircle },
                { label: "Success Rate", value: "87.5%", icon: CheckCircle },
                { label: "Most Affected", value: "Tomato", icon: Leaf }
              ].map((kpi, i) => (
                <div key={i} className={`${cardBg} p-6 rounded-xl border ${borderClass} hover:shadow-lg transition-all`}>
                  <div className="p-3 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 inline-block mb-4 shadow-md">
                    <kpi.icon className="w-8 h-8 text-white" />
                  </div>
                  <div className="text-3xl font-bold mb-1">{kpi.value}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">{kpi.label}</div>
                </div>
              ))}
            </div>

            <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-lg`}>
              <h3 className="font-semibold text-lg mb-4">Coming soon: Interactive Charts & Reports</h3>
              <p className="text-gray-500">Advanced analytics features are under development.</p>
            </div>
          </div>
        )}

        {/* PROFILE PAGE */}
        {currentPage === 'profile' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className={`${cardBg} p-8 rounded-xl border ${borderClass} shadow-lg`}>
              <div className="flex flex-col md:flex-row items-center gap-6">
                <div className="w-24 h-24 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center text-white text-4xl font-bold shadow-xl">A</div>
                <div className="flex-1 text-center md:text-left">
                  <h2 className="text-3xl font-bold mb-1">Ashish Rathod</h2>
                  <p className="text-gray-500 dark:text-gray-400 mb-3">Farmer ID: F_12345</p>
                  <div className="flex flex-wrap gap-4 justify-center md:justify-start text-sm">
                    <span className="flex items-center gap-1">
                      <MapPin className="w-4 h-4 text-emerald-500" /> Pune, Maharashtra
                    </span>
                    <span className="flex items-center gap-1">
                      <Leaf className="w-4 h-4 text-emerald-500" /> Crops: Tomato, Potato
                    </span>
                    <span className="flex items-center gap-1">
                      <Calendar className="w-4 h-4 text-emerald-500" /> Member since Nov 2024
                    </span>
                  </div>
                </div>
                <button className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-lg hover:from-emerald-600 hover:to-emerald-700 flex items-center gap-2 font-medium shadow-lg hover:shadow-xl transition-all">
                  <Edit className="w-4 h-4" /> Edit Profile
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: "Total Scans", value: "248", icon: Camera },
                { label: "Diseases Found", value: "12", icon: AlertCircle },
                { label: "Treatments", value: "34", icon: CheckCircle },
                { label: "Success Rate", value: "89%", icon: TrendingUp }
              ].map((stat, i) => (
                <div key={i} className={`${cardBg} p-6 rounded-xl border ${borderClass} hover:shadow-lg transition-all`}>
                  <stat.icon className="w-8 h-8 text-emerald-500 mb-3" />
                  <div className="text-2xl font-bold mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SETTINGS PAGE */}
        {currentPage === 'settings' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="mb-8">
              <h1 className="text-3xl md:text-4xl font-bold mb-2 bg-gradient-to-r from-emerald-600 to-emerald-500 bg-clip-text text-transparent">
                Settings
              </h1>
              <p className="text-gray-500 dark:text-gray-400">Manage your preferences and account settings</p>
            </div>

            <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-lg`}>
              <h3 className="font-semibold text-lg mb-4">General</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between py-3">
                  <div>
                    <div className="font-medium">Theme</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">Choose your display theme</div>
                  </div>
                  <button onClick={toggleTheme} className={`px-6 py-2 border ${borderClass} rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 font-medium transition-colors shadow-sm`}>
                    {theme === 'light' ? '☀️ Light' : '🌙 Dark'}
                  </button>
                </div>
                <div className={`border-t ${borderClass} pt-4`}>
                  <div className="font-medium mb-2">Language</div>
                  <select className={`w-full px-4 py-2 ${cardBg} border ${borderClass} rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 shadow-sm`}>
                    <option>English</option>
                    <option>हिन्दी (Hindi)</option>
                    <option>मराठी (Marathi)</option>
                    <option>ਪੰਜਾਬੀ (Punjabi)</option>
                  </select>
                </div>
              </div>
            </div>

            <div className={`${cardBg} p-6 rounded-xl border ${borderClass} shadow-lg`}>
              <h3 className="font-semibold text-lg mb-4">About</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between py-2">
                  <span className="text-gray-500 dark:text-gray-400">App Version</span>
                  <span className="font-medium">1.0.0</span>
                </div>
                <div className="flex justify-between py-2">
                  <span className="text-gray-500 dark:text-gray-400">Model Version</span>
                  <span className="font-medium">EfficientNetB0-v1</span>
                </div>
                <div className="flex justify-between py-2">
                  <span className="text-gray-500 dark:text-gray-400">Model Accuracy</span>
                  <span className="font-medium text-emerald-600">95.3%</span>
                </div>
                <div className="flex justify-between py-2">
                  <span className="text-gray-500 dark:text-gray-400">Inference Speed</span>
                  <span className="font-medium">~80ms</span>
                </div>
              </div>
              <div className={`flex flex-col sm:flex-row gap-3 mt-6 pt-6 border-t ${borderClass}`}>
                <button className={`flex-1 px-4 py-2 border ${borderClass} rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors font-medium`}>
                  Terms of Service
                </button>
                <button className={`flex-1 px-4 py-2 border ${borderClass} rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors font-medium`}>
                  Privacy Policy
                </button>
              </div>
              <button className="w-full mt-3 px-4 py-3 text-red-600 border-2 border-red-300 dark:border-red-800 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center justify-center gap-2 font-medium transition-colors shadow-sm">
                <LogOut className="w-4 h-4" /> Logout
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

