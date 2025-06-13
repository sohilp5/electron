// src/_pages/Solutions.tsx
import React, { useState, useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

import ScreenshotQueue from "../components/Queue/ScreenshotQueue";
import { ProblemStatementData } from "../types/solutions";
import SolutionCommands from "../components/Solutions/SolutionCommands";
import Debug from "./Debug";
import { useToast } from "../contexts/toast";
import { COMMAND_KEY } from "../utils/platform";

import { Marked } from 'marked';
// import katex from 'katex'; // Already imported by marked-katex-extension
import 'katex/dist/katex.min.css';
import markedKatex from 'marked-katex-extension';

const katexExtensionOptions = {
  throwOnError: false,
  delimiters: [
    { left: "$$", right: "$$", display: true },
    { left: "$", right: "$", display: false },
    { left: "\\(", right: "\\)", display: false },
    { left: "\\[", right: "\\]", display: true }
  ],
};

const customMarked = new Marked();
customMarked.use(markedKatex(katexExtensionOptions));

// MODIFIED SolutionPayload interface
interface SolutionPayload {
  code: string;
  walkthrough: string; // Was string[] | undefined, now string (Markdown content)
  reflection: string;  // Was thoughts: string[], now string (Markdown content) & renamed
  time_complexity: string;
  space_complexity: string;
}

const getSyntaxHighlighterLanguage = (lang: string): string => {
  switch (lang?.toLowerCase()) {
    case "javascript": return "jsx";
    case "python": return "python";
    case "java": return "java";
    case "c++": case "cpp": return "cpp";
    case "c#": case "csharp": return "csharp";
    case "golang": return "go";
    case "ruby": return "ruby";
    case "swift": return "swift";
    case "kotlin": return "kotlin";
    case "sql": case "oraclesql": return "sql";
    case "r": return "r";
    default: return "plaintext";
  }
};

const extractActualCode = (codeString: string | null): string => {
  if (!codeString) return "// Code not available";
  
  // Since the backend now aims to provide clean code via regex,
  // this function acts more as a fallback or for other contexts if needed.
  // For the main solutionCode, it should ideally be the direct output from the backend.
  const trimmedCodeString = codeString.trim();

  // Priority 1: If it's already clean code (most likely scenario now)
  // A simple heuristic: if it doesn't start with typical Markdown/prose and contains newlines or common code chars.
  if (!trimmedCodeString.startsWith("#") && !trimmedCodeString.startsWith("```") && (trimmedCodeString.includes('\n') || trimmedCodeString.match(/[;{}=()<>]|\b(def|class|function|public|private|import|select|from|where)\b/i))) {
    // Check if it's not just a single line of prose that happens to have a semicolon
     if (trimmedCodeString.length > 10 || trimmedCodeString.includes('\n')) {
        return trimmedCodeString;
     }
  }

  // Priority 2: Extract from Markdown code block (if somehow it's still wrapped)
  const codeBlockRegex = /```(?:[a-zA-Z0-9_.-]*\s*\n)?([\s\S]*?)\n?```/;
  const match = trimmedCodeString.match(codeBlockRegex);

  if (match && match[1]) {
    return match[1].trim(); 
  }
  
  // Priority 3: Strip known preambles if no code block is found
  const preambles = [
    "Code:", "Solution:", "Here's the code:", "Here is the code:",
    "Optimized implementation in python:", "Optimized implementation in java:", // etc.
  ];

  for (const preamble of preambles) {
    if (trimmedCodeString.toLowerCase().startsWith(preamble.toLowerCase())) {
      const potentialCode = trimmedCodeString.substring(preamble.length).trim();
      if (potentialCode.includes('\n') || potentialCode.length > 20) {
        return potentialCode;
      }
    }
  }
  
  // Final fallback:
  if (trimmedCodeString.length < 20 && !trimmedCodeString.includes('\n') && !trimmedCodeString.startsWith("//")) {
      return `// Code not found or unformatted. Original: "${trimmedCodeString}"`;
  }
  return trimmedCodeString; // Return as is, it might be correctly formatted code already
};


export const ContentSection = ({
  title,
  content,
  isLoading,
  isWalkthrough // To control max-width for better readability of long text
}: {
  title: string;
  content: React.ReactNode | string; // content is now expected to be a string for Markdown
  isLoading: boolean;
  isWalkthrough?: boolean;
}) => (
  <div className="space-y-2">
    <h2 className="text-[13px] font-medium text-white tracking-wide">
      {title}
    </h2>
    {isLoading ? (
      <div className="mt-4 flex">
        <p className="text-xs bg-gradient-to-r from-gray-300 via-gray-100 to-gray-300 bg-clip-text text-transparent animate-pulse">
          {title === "Problem Statement" ? "Extracting problem statement..." : "Loading content..."}
        </p>
      </div>
    ) : (
      <>
        {typeof content === 'string' && content.trim() !== "" ? (
          <div
            className={`prose prose-sm prose-invert text-[13px] leading-snug text-gray-100 ${isWalkthrough ? 'max-w-full' : 'max-w-[600px]'} break-words`} // Added break-words
            dangerouslySetInnerHTML={{ __html: customMarked.parse(content) as string }}
          />
        ) : typeof content !== 'string' && content ? ( 
          <div className={`text-[13px] leading-snug text-gray-100 ${isWalkthrough ? 'max-w-full' : 'max-w-[600px]'}`}>
            {content}
          </div>
        ) : (
           <p className="text-xs text-gray-400 italic">Content not available for this section.</p> 
        )}
      </>
    )}
  </div>
);

const SolutionSection = ({
  title,
  content, 
  isLoading,
  currentLanguage,
  showToast // Added showToast prop
}: {
  title: string;
  content: string | null; 
  isLoading: boolean;
  currentLanguage: string;
  showToast: (title: string, message: string, type: "success" | "error" | "neutral") => void; // Added showToast prop type
}) => {
  const [copied, setCopied] = useState(false);
  // The 'content' for SolutionSection should be the direct code string from backend
  const actualCode = content || "// Code not available";


  const copyToClipboard = () => {
    if (actualCode && !actualCode.startsWith("// Code not")) {
      // Attempt to use navigator.clipboard.writeText first
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(actualCode).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        }).catch(err => {
          console.warn("navigator.clipboard.writeText failed, falling back to execCommand:", err);
          fallbackCopyToClipboard(actualCode);
        });
      } else {
        // Fallback for environments where navigator.clipboard is not available (e.g., insecure contexts)
        fallbackCopyToClipboard(actualCode);
      }
    }
  };

  const fallbackCopyToClipboard = (text: string) => {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    // Avoid scrolling to bottom
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      const successful = document.execCommand('copy');
      if (successful) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        console.error('Fallback: Copying text command was unsuccessful');
        showToast("Copy Failed", "Could not copy code to clipboard.", "error"); // Now uses prop
      }
    } catch (err) {
      console.error('Fallback: Oops, unable to copy', err);
      showToast("Copy Failed", "Could not copy code to clipboard.", "error"); // Now uses prop
    }
    document.body.removeChild(textArea);
  };

  return (
    <div className="space-y-2 relative">
      <h2 className="text-[13px] font-medium text-white tracking-wide">
        {title}
      </h2>
      {isLoading ? (
        <div className="space-y-1.5">
          <div className="mt-4 flex">
            <p className="text-xs bg-gradient-to-r from-gray-300 via-gray-100 to-gray-300 bg-clip-text text-transparent animate-pulse">
              Loading solution code...
            </p>
          </div>
        </div>
      ) : (
        <div className="w-full relative">
          <button
            onClick={copyToClipboard}
            className="absolute top-2 right-2 text-xs text-white bg-white/10 hover:bg-white/20 rounded px-2 py-1 transition z-10"
            disabled={actualCode.startsWith("// Code not")}
          >
            {copied ? "Copied!" : "Copy"}
          </button>
          <SyntaxHighlighter
            showLineNumbers
            language={getSyntaxHighlighterLanguage(currentLanguage)}
            style={dracula}
            customStyle={{
              maxWidth: "100%",
              margin: 0,
              padding: "1rem",
              paddingTop: "2.5rem", 
              whiteSpace: "pre-wrap",
              wordBreak: "break-all",
              backgroundColor: "rgba(22, 27, 34, 0.5)", // Slightly transparent background
              borderRadius: "0.375rem", // Tailwind's rounded-md
            }}
            lineProps={{style: {wordBreak: 'break-all', whiteSpace: 'pre-wrap'}}}
            wrapLines={true}
            // wrapLongLines={true} // `wrapLines` should handle this with `pre-wrap`
          >
            {actualCode}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  );
};

export const ComplexitySection = ({ 
  timeComplexity,
  spaceComplexity,
  isLoading
}: {
  timeComplexity: string | null
  spaceComplexity: string | null
  isLoading: boolean
}) => {
  // Helper to ensure Big O notation is present and format explanation
  const formatComplexityDetail = (detail: string | null, type: 'Time' | 'Space'): React.ReactNode => {
    if (!detail || detail.trim() === "" || detail.includes("not found") || detail.includes("N/A")) {
        return `Complexity analysis for ${type.toLowerCase()} not available.`;
    }

    const bigORegex = /(O\([^)]+\))/i;
    const match = detail.match(bigORegex);
    
    let notation = "";
    let explanation = detail;

    if (match) {
        notation = match[0];
        explanation = detail.replace(bigORegex, '').replace(/^[\s-]+/, '').trim();
        if (!explanation) explanation = "No detailed explanation provided.";
    } else {
        // If no O() notation found, assume the whole string is an explanation or a placeholder
        // Try to infer O() if common terms are used
        if (detail.toLowerCase().includes("linear")) notation = "O(n)";
        else if (detail.toLowerCase().includes("quadratic")) notation = "O(n^2)";
        else if (detail.toLowerCase().includes("logarithmic")) notation = "O(log n)";
        else if (detail.toLowerCase().includes("constant")) notation = "O(1)";
        // If still no notation, the original detail will be shown as explanation
    }
    
    // Render using dangerouslySetInnerHTML to process Markdown/KaTeX in explanation
    const parsedExplanation = customMarked.parse(explanation) as string;

    return (
      <>
        {notation && <strong className="mr-1">{notation}</strong>}
        <span dangerouslySetInnerHTML={{ __html: parsedExplanation }} />
      </>
    );
};
  
  return (
    <div className="space-y-2">
      <h2 className="text-[13px] font-medium text-white tracking-wide">Complexity</h2>
      {isLoading ? (
        <p className="text-xs bg-gradient-to-r from-gray-300 via-gray-100 to-gray-300 bg-clip-text text-transparent animate-pulse">
          Calculating complexity...
        </p>
      ) : (
        <div className="space-y-3">
          <div className="text-[13px] leading-snug text-gray-100 bg-white/5 rounded-md p-3">
            <div className="flex items-start gap-2">
              <div className="w-1 h-1 rounded-full bg-blue-400/80 mt-1.5 shrink-0" />
              <div><strong>Time: </strong>{formatComplexityDetail(timeComplexity, 'Time')}</div>
            </div>
          </div>
          <div className="text-[13px] leading-snug text-gray-100 bg-white/5 rounded-md p-3">
            <div className="flex items-start gap-2">
              <div className="w-1 h-1 rounded-full bg-green-400/80 mt-1.5 shrink-0" /> {/* Changed color for distinction */}
              <div><strong>Space: </strong>{formatComplexityDetail(spaceComplexity, 'Space')}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};


export interface SolutionsProps {
  setView: (view: "queue" | "solutions" | "debug") => void;
  credits: number;
  currentLanguage: string;
  setLanguage: (language: string) => void;
}

const Solutions: React.FC<SolutionsProps> = ({
  setView,
  credits,
  currentLanguage,
  setLanguage
}) => {
  const queryClient = useQueryClient();
  const contentRef = useRef<HTMLDivElement>(null);
  const { showToast } = useToast(); // Assuming useToast is correctly set up

  const [debugProcessing, setDebugProcessing] = useState(false);
  const [problemStatementData, setProblemStatementData] = useState<ProblemStatementData | null>(null);
  const [solutionCode, setSolutionCode] = useState<string | null>(null); 
  // MODIFIED state variables for walkthrough and reflection
  const [walkthroughData, setWalkthroughData] = useState<string | null>(null);
  const [reflectionData, setReflectionData] = useState<string | null>(null); // Renamed from thoughtsData
  
  const [timeComplexityData, setTimeComplexityData] = useState<string | null>(null);
  const [spaceComplexityData, setSpaceComplexityData] = useState<string | null>(null);

  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const [tooltipHeight, setTooltipHeight] = useState(0);
  const [isResetting, setIsResetting] = useState(false);

  interface Screenshot {
    id: string;
    path: string;
    preview: string;
    timestamp: number;
  }
  const [extraScreenshots, setExtraScreenshots] = useState<Screenshot[]>([]);


  useEffect(() => {
    const fetchScreenshots = async () => {
        try {
            const existingScreenshotsData = await window.electronAPI.getScreenshots();
            let screenshotsToSet: Screenshot[] = [];
            if (Array.isArray(existingScreenshotsData)) {
                screenshotsToSet = existingScreenshotsData.map((p: any) => ({ id: p.path, path: p.path, preview: p.preview, timestamp: Date.now() }));
            } else if (existingScreenshotsData && Array.isArray(existingScreenshotsData.previews)) {
                screenshotsToSet = existingScreenshotsData.previews.map((p: { path: string; preview: string; }) => ({ id: p.path, path: p.path, preview: p.preview, timestamp: Date.now() }));
            }
            setExtraScreenshots(screenshotsToSet);
        } catch (error) { console.error("Error loading extra screenshots:", error); setExtraScreenshots([]); }
    };
    if (solutionCode) fetchScreenshots();
  }, [solutionCode]);

  useEffect(() => {
    const cleanupFunctions = [
      window.electronAPI.onScreenshotTaken(async () => {
        try {
            const existingScreenshotsData = await window.electronAPI.getScreenshots();
            let screenshotsToSet: Screenshot[] = [];
             if (Array.isArray(existingScreenshotsData)) { 
                screenshotsToSet = existingScreenshotsData.map((p: any) => ({ id: p.path, path: p.path, preview: p.preview, timestamp: Date.now() }));
            } else if (existingScreenshotsData && Array.isArray(existingScreenshotsData.previews)) {
                screenshotsToSet = existingScreenshotsData.previews.map((p: { path: string; preview: string; }) => ({ id: p.path, path: p.path, preview: p.preview, timestamp: Date.now() }));
            }
            setExtraScreenshots(screenshotsToSet);
        } catch (error) { console.error("Error re-fetching extra screenshots:", error); }
      }),
      window.electronAPI.onResetView(() => {
        setIsResetting(true);
        queryClient.setQueryData(["solution"], null); 
        queryClient.setQueryData(["new_solution"], null);
        queryClient.setQueryData(["problem_statement"], null);
        setExtraScreenshots([]);
        setProblemStatementData(null); 
        setSolutionCode(null);
        // MODIFIED reset
        setWalkthroughData(null);
        setReflectionData(null); // Renamed
        setTimeComplexityData(null);
        setSpaceComplexityData(null);
        setTimeout(() => setIsResetting(false), 0);
      }),
      window.electronAPI.onSolutionStart(() => {
        setSolutionCode(null); 
        // MODIFIED start
        setWalkthroughData(null); 
        setReflectionData(null); // Renamed
        setTimeComplexityData(null); 
        setSpaceComplexityData(null);
      }),
      window.electronAPI.onProblemExtracted((data: ProblemStatementData) => {
        queryClient.setQueryData(["problem_statement"], data);
        setProblemStatementData(data); 
      }),
      window.electronAPI.onSolutionError((error: string) => {
        showToast("Processing Failed", error, "error");
        const solution = queryClient.getQueryData(["solution"]) as SolutionPayload | null;
        if (!solution && !problemStatementData) { 
          setView("queue");
        }
        setSolutionCode(solution?.code || null);
        // MODIFIED error handling
        setWalkthroughData(solution?.walkthrough || null);
        setReflectionData(solution?.reflection || null); // Renamed
        setTimeComplexityData(solution?.time_complexity || null);
        setSpaceComplexityData(solution?.space_complexity || null);
      }),
      window.electronAPI.onSolutionSuccess((data: SolutionPayload) => { // Ensure data type matches
        if (!data) { console.warn("Received empty solution data"); return; }
        queryClient.setQueryData(["solution"], data);
        setSolutionCode(data.code || null);
        // MODIFIED success handling
        setWalkthroughData(data.walkthrough || null); // data.walkthrough is now a string
        setReflectionData(data.reflection || null);   // data.reflection is now a string
        setTimeComplexityData(data.time_complexity || null);
        setSpaceComplexityData(data.space_complexity || null);
      }),
      window.electronAPI.onDebugStart(() => setDebugProcessing(true)),
      window.electronAPI.onDebugSuccess((data: any) => { // Type 'any' for debug data for now
        queryClient.setQueryData(["new_solution"], data);
        setDebugProcessing(false);
      }),
      window.electronAPI.onDebugError(() => {
        showToast("Processing Failed", "There was an error debugging your code.", "error");
        setDebugProcessing(false);
      }),
      window.electronAPI.onProcessingNoScreenshots(() => {
        showToast("No Screenshots", "There are no extra screenshots to process.", "neutral");
      }),
    ];
    return () => cleanupFunctions.forEach((cleanup) => cleanup());
  }, [queryClient, setView, showToast, problemStatementData]); // Added problemStatementData to deps

  useEffect(() => {
    const updateDimensions = () => {
      if (contentRef.current) {
        let contentHeight = contentRef.current.scrollHeight;
        const contentWidth = contentRef.current.scrollWidth;
        if (isTooltipVisible) {
          contentHeight += tooltipHeight;
        }
        window.electronAPI.updateContentDimensions({
          width: contentWidth || 800, 
          height: contentHeight || 600, 
        });
      }
    };
    updateDimensions(); 
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (contentRef.current) {
      resizeObserver.observe(contentRef.current);
    }
    const timeoutId = setTimeout(updateDimensions, 100); 
    return () => {
        resizeObserver.disconnect();
        clearTimeout(timeoutId);
    }
    // MODIFIED dependencies for resize effect
  }, [isTooltipVisible, tooltipHeight, problemStatementData, solutionCode, walkthroughData, reflectionData, timeComplexityData, spaceComplexityData]);

  useEffect(() => {
    const cachedProblemStatement = queryClient.getQueryData(["problem_statement"]) as ProblemStatementData | null;
    if (cachedProblemStatement) setProblemStatementData(cachedProblemStatement);

    const cachedSolution = queryClient.getQueryData(["solution"]) as SolutionPayload | null;
    if (cachedSolution) {
        setSolutionCode(cachedSolution.code ?? null);
        // MODIFIED cache loading
        setWalkthroughData(cachedSolution.walkthrough ?? null);
        setReflectionData(cachedSolution.reflection ?? null); // Renamed
        setTimeComplexityData(cachedSolution.time_complexity ?? null);
        setSpaceComplexityData(cachedSolution.space_complexity ?? null);
    }

    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (event?.query.queryKey[0] === "problem_statement") {
        setProblemStatementData(queryClient.getQueryData(["problem_statement"]) as ProblemStatementData || null);
      }
      if (event?.query.queryKey[0] === "solution") {
        const currentSolution = queryClient.getQueryData(["solution"]) as SolutionPayload | null;
        setSolutionCode(currentSolution?.code ?? null);
        // MODIFIED cache subscription
        setWalkthroughData(currentSolution?.walkthrough ?? null);
        setReflectionData(currentSolution?.reflection ?? null); // Renamed
        setTimeComplexityData(currentSolution?.time_complexity ?? null);
        setSpaceComplexityData(currentSolution?.space_complexity ?? null);
      }
    });
    return () => unsubscribe();
  }, [queryClient]);

  const handleTooltipVisibilityChange = (visible: boolean, height: number) => {
    setIsTooltipVisible(visible);
    setTooltipHeight(height);
  };

  const handleDeleteExtraScreenshot = async (index: number) => {
    const screenshotToDelete = extraScreenshots[index];
    try {
      const response = await window.electronAPI.deleteScreenshot(screenshotToDelete.path);
      if (response.success) {
        const existingScreenshotsData = await window.electronAPI.getScreenshots();
        let screenshotsToSet: Screenshot[] = [];
        if (Array.isArray(existingScreenshotsData)) { 
            screenshotsToSet = existingScreenshotsData.map((p: any) => ({ id: p.path, path: p.path, preview: p.preview, timestamp: Date.now() }));
        } else if (existingScreenshotsData && Array.isArray(existingScreenshotsData.previews)) {
            screenshotsToSet = existingScreenshotsData.previews.map((p: { path: string; preview: string; }) => ({ id: p.path, path: p.path, preview: p.preview, timestamp: Date.now() }));
        }
        setExtraScreenshots(screenshotsToSet);
      } else {
        showToast("Error", response.error || "Failed to delete the screenshot", "error");
      }
    } catch (error: any) {
      showToast("Error", error.message || "Failed to delete the screenshot", "error");
    }
  };
  
  // walkthroughData and reflectionData are now strings, no need for specific contentString processing here
  // They will be passed directly to ContentSection

  const isLoadingDisplay = !problemStatementData || (!solutionCode && !queryClient.getQueryData(["new_solution"]));


  return (
    <>
      {!isResetting && queryClient.getQueryData(["new_solution"]) ? (
        <Debug
          isProcessing={debugProcessing}
          setIsProcessing={setDebugProcessing}
          currentLanguage={currentLanguage}
          setLanguage={setLanguage}
        />
      ) : (
        // <div ref={contentRef} className="relative w-full"> 
        <div className="relative h-screen overflow-y-auto bg-black/70 backdrop-blur-sm">
          <div className="space-y-3 px-4 py-3">
            {solutionCode && extraScreenshots.length > 0 && ( 
              <div className="bg-transparent w-fit">
                <div className="pb-3">
                  <div className="space-y-3 w-fit">
                    <ScreenshotQueue
                      isLoading={debugProcessing}
                      screenshots={extraScreenshots}
                      onDeleteScreenshot={handleDeleteExtraScreenshot}
                    />
                  </div>
                </div>
              </div>
            )}

            <SolutionCommands
              onTooltipVisibilityChange={handleTooltipVisibilityChange}
              isProcessing={isLoadingDisplay && !debugProcessing} 
              extraScreenshots={extraScreenshots}
              credits={credits}
              currentLanguage={currentLanguage}
              setLanguage={setLanguage}
            />

            <div className="w-full text-sm text-black bg-black/60 rounded-md">
              <div className="rounded-lg overflow-hidden">
                <div className="px-4 py-3 space-y-6 max-w-full">
                  {isLoadingDisplay && !problemStatementData && !solutionCode && (
                     <div className="text-center py-10">
                        <p className="text-gray-400">Take screenshots of a coding problem and process them.</p>
                    </div>
                  )}
                  
                  {problemStatementData && (
                     <ContentSection
                        title="Problem Statement"
                        content={problemStatementData.problem_statement}
                        isLoading={!problemStatementData && isLoadingDisplay} 
                      />
                  )}

                  {problemStatementData && !solutionCode && isLoadingDisplay && (
                    <div className="mt-4 flex">
                      <p className="text-xs bg-gradient-to-r from-gray-300 via-gray-100 to-gray-300 bg-clip-text text-transparent animate-pulse">
                        Generating solutions...
                      </p>
                    </div>
                  )}

                  {/* MODIFIED rendering order and content */}
                  {solutionCode && problemStatementData && (
                    <>
                      <ContentSection
                        title="Thought Process / Walkthrough"
                        isWalkthrough // For max-width styling
                        content={walkthroughData || "No detailed walkthrough provided."} // Pass string directly
                        isLoading={isLoadingDisplay && !walkthroughData}
                      />

                      <SolutionSection
                        title="Solution Code"
                        content={solutionCode} 
                        isLoading={!solutionCode && isLoadingDisplay}
                        currentLanguage={currentLanguage}
                        showToast={showToast} // Pass showToast here
                      />
                      
                      <ContentSection
                        title="Post-Solution Reflection (Thoughts)"
                        content={reflectionData || "No specific post-solution reflection provided."} // Pass string directly
                        isLoading={isLoadingDisplay && !reflectionData}
                      />

                      <ComplexitySection
                        timeComplexity={timeComplexityData}
                        spaceComplexity={spaceComplexityData}
                        isLoading={(!timeComplexityData || !spaceComplexityData) && isLoadingDisplay}
                      />
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Solutions;
