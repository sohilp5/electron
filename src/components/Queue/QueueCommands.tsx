import React, { useState, useEffect, useRef } from "react"
import { createRoot } from "react-dom/client"

import { useToast } from "../../contexts/toast"
import { LanguageSelector } from "../shared/LanguageSelector"
import { COMMAND_KEY } from "../../utils/platform"

interface QueueCommandsProps {
  onTooltipVisibilityChange: (visible: boolean, height: number) => void
  screenshotCount?: number
  credits: number
  currentLanguage: string
  setLanguage: (language: string) => void
  // Add props for the click handlers
  onSolve: () => void
  onSolveGeneral: () => void
}

const QueueCommands: React.FC<QueueCommandsProps> = ({
  onTooltipVisibilityChange,
  screenshotCount = 0,
  credits,
  currentLanguage,
  setLanguage,
  onSolve,
  onSolveGeneral
}) => {
  const [isTooltipVisible, setIsTooltipVisible] = useState(false)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const { showToast } = useToast()

  // This function can remain as it is
  const extractLanguagesAndUpdate = (direction?: 'next' | 'prev') => {
    const hiddenRenderContainer = document.createElement('div');
    hiddenRenderContainer.style.position = 'absolute';
    hiddenRenderContainer.style.left = '-9999px';
    document.body.appendChild(hiddenRenderContainer);
    
    const root = createRoot(hiddenRenderContainer);
    root.render(
      <LanguageSelector 
        currentLanguage={currentLanguage} 
        setLanguage={() => {}}
      />
    );
    
    setTimeout(() => {
      const selectElement = hiddenRenderContainer.querySelector('select');
      if (selectElement) {
        const options = Array.from(selectElement.options);
        const values = options.map(opt => opt.value);
        
        const currentIndex = values.indexOf(currentLanguage);
        let newIndex = currentIndex;
        
        if (direction === 'prev') {
          newIndex = (currentIndex - 1 + values.length) % values.length;
        } else {
          newIndex = (currentIndex + 1) % values.length;
        }
        
        if (newIndex !== currentIndex) {
          setLanguage(values[newIndex]);
          window.electronAPI.updateConfig({ language: values[newIndex] });
        }
      }
      
      root.unmount();
      document.body.removeChild(hiddenRenderContainer);
    }, 50);
  };

  useEffect(() => {
    let tooltipHeight = 0
    if (tooltipRef.current && isTooltipVisible) {
      tooltipHeight = tooltipRef.current.offsetHeight + 10
    }
    onTooltipVisibilityChange(isTooltipVisible, tooltipHeight)
  }, [isTooltipVisible, onTooltipVisibilityChange])

  const handleSignOut = async () => {
    try {
      localStorage.clear();
      sessionStorage.clear();
      
      await window.electronAPI.updateConfig({
        apiKey: '',
      });
      
      showToast('Success', 'Logged out successfully', 'success');
      
      setTimeout(() => {
        window.location.reload();
      }, 1500);
    } catch (err) {
      console.error("Error logging out:", err);
      showToast('Error', 'Failed to log out', 'error');
    }
  }

  const handleMouseEnter = () => {
    setIsTooltipVisible(true)
  }

  const handleMouseLeave = () => {
    setIsTooltipVisible(false)
  }

  return (
    <div>
      <div className="pt-2 w-fit">
        <div className="text-xs text-white/90 backdrop-blur-md bg-black/60 rounded-lg py-2 px-4 flex items-center justify-center gap-4">
          {/* Screenshot */}
          <div
            className="flex items-center gap-2 cursor-pointer rounded px-2 py-1.5 hover:bg-white/10 transition-colors"
            onClick={async () => {
              try {
                const result = await window.electronAPI.triggerScreenshot()
                if (!result.success) {
                  console.error("Failed to take screenshot:", result.error)
                  showToast("Error", "Failed to take screenshot", "error")
                }
              } catch (error) {
                console.error("Error taking screenshot:", error)
                showToast("Error", "Failed to take screenshot", "error")
              }
            }}
          >
            <span className="text-[11px] leading-none truncate">
              {screenshotCount === 0
                ? "Take first screenshot"
                : screenshotCount === 1
                ? "Take second screenshot"
                : screenshotCount === 2
                ? "Take third screenshot"
                : screenshotCount === 3
                ? "Take fourth screenshot"
                : screenshotCount === 4
                ? "Take fifth screenshot"
                : "Next will replace first screenshot"}
            </span>
            <div className="flex gap-1">
              <button className="bg-white/10 rounded-md px-1.5 py-1 text-[11px] leading-none text-white/70">
                {COMMAND_KEY}
              </button>
              <button className="bg-white/10 rounded-md px-1.5 py-1 text-[11px] leading-none text-white/70">
                H
              </button>
            </div>
          </div>

          {/* Solve Coding Problem Command - MODIFIED */}
          {screenshotCount > 0 && (
            <div
              className={`flex flex-col cursor-pointer rounded px-2 py-1.5 hover:bg-white/10 transition-colors`}
              onClick={onSolve} // Use the prop here
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] leading-none">Solve Code</span>
                <div className="flex gap-1 ml-2">
                  <button className="bg-white/10 rounded-md px-1.5 py-1 text-[11px] leading-none text-white/70">
                    {COMMAND_KEY}
                  </button>
                  <button className="bg-white/10 rounded-md px-1.5 py-1 text-[11px] leading-none text-white/70">
                    ↵
                  </button>
                </div>
              </div>
            </div>
          )}
          {/* Solve General Problem Button - MODIFIED */}
          {screenshotCount > 0 && (
            <div
              className={`flex flex-col cursor-pointer rounded px-2 py-1.5 hover:bg-white/10 transition-colors`}
              onClick={onSolveGeneral} // Use the prop here
              title="Solve as a general problem (math, case study, etc.)"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] leading-none">Solve General</span>
                <div className="flex gap-1 ml-2">
                   <button className="bg-white/10 rounded-md px-1.5 py-1 text-[11px] leading-none text-white/70">
                    {COMMAND_KEY}
                  </button>
                  <button className="bg-white/10 rounded-md px-1.5 py-1 text-[11px] leading-none text-white/70">
                    G 
                  </button>
                </div>
              </div>
            </div>
          )}
          {/* Separator */}
          <div className="mx-2 h-4 w-px bg-white/20" />

          {/* Settings with Tooltip */}
          <div
            className="relative inline-block"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            {/* Gear icon */}
            <div className="w-4 h-4 flex items-center justify-center cursor-pointer text-white/70 hover:text-white/90 transition-colors">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="w-3.5 h-3.5"
              >
                <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l-.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </div>

            {/* Tooltip Content */}
            {isTooltipVisible && (
              <div
                ref={tooltipRef}
                className="absolute top-full left-0 mt-2 w-80 transform -translate-x-[calc(50%-12px)]"
                style={{ zIndex: 100 }}
              >
                <div className="absolute -top-2 right-0 w-full h-2" />
                <div className="p-3 text-xs bg-black/80 backdrop-blur-md rounded-lg border border-white/10 text-white/90 shadow-lg">
                  <div className="space-y-4">
                    <h3 className="font-medium truncate">Keyboard Shortcuts</h3>
                    <div className="space-y-3">
                      {/* ... (All the shortcut display divs remain unchanged) ... */}
                      <div className="cursor-pointer rounded px-2 py-1.5 hover:bg-white/10 transition-colors">
                        <div className="flex items-center justify-between">
                          <span className="truncate">Toggle Window</span>
                          <div className="flex gap-1 flex-shrink-0"><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">{COMMAND_KEY}</span><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">B</span></div>
                        </div>
                        <p className="text-[10px] leading-relaxed text-white/70 truncate mt-1">Show or hide this window.</p>
                      </div>
                      <div className="cursor-pointer rounded px-2 py-1.5 hover:bg-white/10 transition-colors">
                        <div className="flex items-center justify-between">
                          <span className="truncate">Take Screenshot</span>
                          <div className="flex gap-1 flex-shrink-0"><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">{COMMAND_KEY}</span><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">H</span></div>
                        </div>
                        <p className="text-[10px] leading-relaxed text-white/70 truncate mt-1">Take a screenshot of the problem description.</p>
                      </div>
                      <div className="cursor-pointer rounded px-2 py-1.5 hover:bg-white/10 transition-colors">
                         <div className="flex items-center justify-between">
                           <span className="truncate">Solve</span>
                           <div className="flex gap-1 flex-shrink-0"><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">{COMMAND_KEY}</span><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">↵</span></div>
                         </div>
                         <p className="text-[10px] leading-relaxed text-white/70 truncate mt-1">Generate a solution based on the current problem.</p>
                      </div>
                      <div className="cursor-pointer rounded px-2 py-1.5 hover:bg-white/10 transition-colors">
                        <div className="flex items-center justify-between">
                          <span className="truncate">Delete Last Screenshot</span>
                           <div className="flex gap-1 flex-shrink-0"><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">{COMMAND_KEY}</span><span className="bg-white/20 px-1.5 py-0.5 rounded text-[10px] leading-none">L</span></div>
                        </div>
                        <p className="text-[10px] leading-relaxed text-white/70 truncate mt-1">Remove the most recently taken screenshot.</p>
                      </div>
                    </div>
                    <div className="pt-3 mt-3 border-t border-white/10">
                      <div className="mb-3 px-2">
                         <div className="flex items-center justify-between cursor-pointer hover:bg-white/10 rounded px-2 py-1 transition-colors" onClick={() => extractLanguagesAndUpdate('next')} tabIndex={0} onKeyDown={(e) => { if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { extractLanguagesAndUpdate('prev'); } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { extractLanguagesAndUpdate('next'); }}}>
                           <span className="text-[11px] text-white/70">Language</span>
                           <div className="flex items-center gap-2">
                             <span className="text-[11px] text-white/90">{currentLanguage}</span>
                             <div className="text-white/40 text-[8px]"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-3 h-3"><path d="M7 13l5 5 5-5M7 6l5 5 5-5"/></svg></div>
                           </div>
                         </div>
                      </div>
                      <div className="mb-3 px-2 space-y-1">
                        <div className="flex items-center justify-between text-[13px] font-medium text-white/90">
                          <span>API Settings</span>
                          <button className="bg-white/10 hover:bg-white/20 px-2 py-1 rounded text-[11px]" onClick={() => window.electronAPI.openSettingsPortal()}>Settings</button>
                        </div>
                      </div>
                      <button onClick={handleSignOut} className="flex items-center gap-2 text-[11px] text-red-400 hover:text-red-300 transition-colors w-full">
                        <div className="w-4 h-4 flex items-center justify-center"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-3 h-3"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" /></svg></div>
                        Log Out
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default QueueCommands