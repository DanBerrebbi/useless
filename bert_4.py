import bert_poids_class
import bert_class
import time
from scipy.spatial.distance import cosine

texte="""Mayor Bill de Blasio’s counsel and chief legal adviser, Maya Wiley, is resigning next month from her City Hall position to become the chairwoman of the Civilian Complaint Review Board, New York City’s independent oversight agency for the Police Department.
The move represents the latest shake-up for the de Blasio administration amid continuing state and federal investigations into the mayor’s fund-raising, and fills a two-month vacancy at the police review board created by the resignation of its chairman, Richard D. Emery, in April.
A civil rights lawyer and advocate for racial and social justice, Ms. Wiley joined the de Blasio administration in early 2014 to focus on legal issues as well as on the mayor’s efforts to address issues of inequality. But over time, Ms. Wiley became discouraged over not being part of Mr. de Blasio’s inner circle and felt cut out of both legal questions and advocacy, according to a person familiar with her thinking. On the former, Mr. de Blasio often relied instead on the city’s corporation counsel and Henry Berger, the mayor’s special counsel; on the latter, he favored his top political aides. The person requested anonymity to discuss private conversations.
More recently, Ms. Wiley was assigned to help craft the administration’s legal response to the state and federal inquiries as well as to requests for the public disclosure of documents, notably emails between Mr. de Blasio and trusted advisers outside the administration.
It was in response to a question from reporters about the withholding of those emails with advisers that Ms. Wiley, defending the practice, described the advisers as “agents of the city” — a designation that appeared novel and resulted in days of unfavorable press coverage.
In a statement on Wednesday, Mr. de Blasio thanked Ms. Wiley for her service and congratulated her on her new role.
The review board investigates allegations of misconduct by officers and makes recommendations for discipline to the Police Department. Its data on the number of complaints, and its investigations of officers, provide an important barometer of police behavior and a politically important one for Mr. de Blasio, a Democrat who campaigned on improving police-community relations.
Mr. Wiley will also take a position at the New School in Manhattan. Her moves were reported by The Wall Street Journal.
The announcement of Ms. Wiley’s departure from City Hall followed that of a recently hired director of social media, Scott Kleinberg, who resigned on Saturday, just eight weeks after being hired to bring greater personality to the Twitter, Facebook and other online accounts associated with the mayor’s office. His resignation was reported by DNA Info.
In a Facebook post that was later removed, Mr. Kleinberg complained of long hours and micromanagement and described his experience with the administration as working with “political hacks plus a boss who just couldn’t get it,” adding, “It was a bad combination for sure.” Mr. Kleinberg declined to comment.
The departures came less than two months after Karen Hinton, the mayor’s top spokeswoman, announced her resignation from the administration. (She stayed in the position until mid-June)."""
texte="A group of people is equipped with protective gear."
t1=time.time()
test=bert_poids_class.keywords(texte)
test.extraction()
test.remplissage()

test_bis=bert_class.article(texte)
test_bis.sent_embeddings()
test_bis.article_embedding(test.poids)

print("Temps 1 : {}".format(time.time()-t1))

texte2="""When the New York City Education Department put a new Success Academy charter school in a building housing a troubled Brooklyn middle school in 2012, many believed the middle school was on its way to closing.
The school, Junior High School 50, known as John D. Wells, had struggled for years. In 2014, Mayor Bill de Blasio, a Democrat, included it on a list of 94 chronically low-performing schools that the city was infusing with money and social services. Only 49 students were admitted last year.
But instead of dying, J.H.S. 50, in the Williamsburg neighborhood, is showing signs of revival.
A new principal has added programs and activities. The school is hopeful about its performance on this year’s standardized tests. And for the first time in many years, enrollment is expected to increase — 112 sixth graders are registered for September — a sign that parents like what they see happening there.
Now, in a twist, even as it grows, J.H.S. 50 will have to give up five classrooms next year, because the Success Academy school is expanding to fifth grade.
Supporters of J.H.S. 50 are accusing the Education Department of betting against a turnaround. Last year, when it approved Success’ expansion, the department drew up a plan assuming that J.H.S. 50’s enrollment would continue to decline, to as few as 165 students next year. The department is now projecting that the school will have around 230 students. The principal, Benjamin Honoroff, believes that enrollment could be as high as 270, given transfers and so-called over-the-counter students — often new immigrants who arrive in the middle of the year and are assigned to J.H.S. 50 because of its transitional bilingual program.
“We doubled our incoming sixth-grade class,” Mr. Honoroff said. “So I think that an equitable allocation would mean reconsidering some of those decisions.”
City Councilman Antonio Reynoso, who graduated from J.H.S. 50 in 1996, when it had close to 1,100 students, said that when the department made the plan, it promised to adjust it if enrollment increased.
“They said yes, that they would do that, and now we’re facing that issue right now and they’re saying no,” Mr. Reynoso, a Democrat whose district includes Williamsburg, said in an interview.
“They assumed that that wasn’t going to happen,” he added.
Success Academy, through a spokesman, disputed the idea that space was unfairly allocated. The spokesman, Stefan Friedman, said that the charter school expected to add 110 students next year, and that even if J.H.S. 50 reached an enrollment of 270, the two schools would have roughly the same number of students per classroom. He said that the department’s space allocations generally disfavored charter schools, and that several other Success schools in the city had many more students per classroom than traditional public schools in the same buildings.
Over the past two decades, as the surrounding area evolved from a neighborhood of bodegas and discount stores, where half the residents received public assistance, into one of multimillion-dollar apartments and popular restaurants, J.H.S. 50 was largely left behind.
Its students are mostly Hispanic and poor. Thirty-one percent have disabilities, and 29 percent do not speak English proficiently. In 2015, soon after J.H.S. 50 became part of Mr. de Blasio’s school improvement effort, which the administration calls Renewal Schools, only 10 percent of the students passed the state reading tests and only 7 percent passed the math test.
While the Renewal program as a whole has had mixed results, there are signs of progress at J.H.S. 50. The school was paired with a community organization called El Puente, which brought in new staff members to work on improving attendance, increasing parent engagement, training students in conflict mediation, and using art to help students in the bilingual program learn English.
Mr. Honoroff, who became principal last year after working as a literacy coach at the school, instituted new math and English curriculums, added small-group reading interventions for all students four times a week, and made room in the schedule for teachers to meet in teams several times a week to examine students’ work. While scores from this spring’s state tests are not back yet, students made substantial improvement on internal reading tests during the year. Mr. Honoroff has also made debate a major focus, and the school’s debate team recently won a citywide competition.
To tackle its low attendance rate, J.H.S. 50 has paired frequently absent students with staff mentors, who call students or visit them at home when they do not show up and offer positive reinforcement when they do. The rate of chronic absenteeism — the share of students whose attendance is below 90 percent — declined to 31 percent in 2015-16 from 40 percent in 2014-15.
J.H.S. 50 is also using the extra hour that all the schools in the Renewal program have added to their day to make going to school more appealing. Students can choose from a wide variety of extracurricular activities, including robotics, video game design, dance, mural painting, rock band and sports like soccer, baseball and basketball.
Mr. Honoroff is worried about losing dedicated space for some of those activities as the school struggles to fit into a smaller footprint next year. J.H.S. 50 will probably have to turn its dance studio into a regular classroom. It is likely to lose a new computer lab Mr. Reynoso financed. And several rooms will need to do double duty, as both a classroom and a music room, for instance.
To be sure, many schools in the city, both public and charter, struggle with space constraints. And the elementary school to which J.H.S. 50 is losing space, Success Academy Williamsburg, performs much better on state tests. Last year, 80 percent of its third graders — then its top grade — passed the reading tests and 99 percent passed the math tests. (The charter school has more white students and middle-class students, and fewer disabled students and students not proficient in English, than J.H.S. 50.)
One morning last week, Mr. Honoroff pointed out five classrooms assigned to Success this past year that he said went unused and three others that he said were used only sporadically, for occupational therapy.
A spokeswoman for the Education Department, Devora Kaye, said that the department was committed to supporting J.H.S. 50’s growth and that it would review space requirements after enrollment numbers were confirmed in October. In the meantime, she said, the department would help the principal with renovations and room conversions to make the space work this year.
But Mr. Reynoso said that having to squeeze into fewer classrooms could hurt the school’s ability to increase enrollment further.
“We’re about to take away space that they were using to attract parents,” he said."""
texte2="The mayor of New-York, Mr. De Blasio will make a report for the coronavirus crisis."
texte2="A group of people is equipped with gear used for protection."
t1=time.time()
test2=bert_poids_class.keywords(texte2)
test2.extraction()
test2.remplissage()

test2_bis=bert_class.article(texte2)
test2_bis.sent_embeddings()
test2_bis.article_embedding(test2.poids)
print("Temps 2 : {}".format(time.time()-t1))

texte3="""Times Insider delivers behind-the-scenes insights from The New York Times. Majd al Waheidi is a Gaza-based “stringer,” a freelance reporter who regularly contributes to The Times. In this piece, she describes a recent — and, for her, a rare — trip to Israel.
As a reporter, I have come to know the Gazan borders very intimately. I once made several reporting trips to a village near the fence for a story about illegal crossings.
But though I live so close to Israel, it is a world away for me.
Last month, I crossed the border and left Gaza for only the third time in my life. I was traveling to Israel to obtain a visa for a summer fellowship in Washington, D.C.
Security on the Gazan side of the border consists of two separate checkpoints: the first (which Gazans call “4/4”) is controlled by Hamas, and the second (“5/5”) is controlled by the Palestinian Authority.
“4/4” and “5/5” are within half a mile of each other, but the two checkpoints represent a lingering division among Palestinians — one that I reported on in 2015.
The passageway between Israel and Gaza is a long tunnel, enclosed by wires and metal. After passing through “4/4” and “5/5,” I was ferried through the tunnel in a tuk-tuk, a small motorized cart with room for two or four people. (My luggage was carried in a separate van.)
When crossing here, at Erez, Gazans often post photos of the tunnel on social media to confirm that they’ve made it this far. I took a picture but chose not to share it; it felt unfair to the Gazans who are unable to leave.
Next came the Israeli terminal, which resembles a large glass box.
After the last war, Israel eased some of their restrictions, and hundreds of Gazans crossed every day. Most of the people who obtained permits were either businessmen or local staff members of international organizations.
Fewer than 10 people were sitting on the chairs waiting for the soldier in the glass enclosure to call each name. Many of them were mothers with their children. Most, I assumed, were patients in need of medical treatment.
They called me a few minutes after I arrived. An Israeli soldier, sitting on a tall chair behind the glass, gave me my permit and my I.D.
It is rare that I see an Israeli in person; Israel withdrew from Gaza in 2005, and I was too young then to remember their presence. However, I witnessed three wars in less than six years. The last one was devastating. I remember the warplanes and F-16s and artillery shells.
At Erez, I saw members of the Israeli defense forces, but not the same ones who fought in wars that left people dead and homes destroyed. I saw Israeli soldiers working administrative jobs, soldiers who help people wanting to leave. I used English with them because I felt they wouldn’t understand me otherwise — though they speak Arabic very well.
Everyone who has passed through the tunnel has told me the same thing: The moment we leave Gaza, the air changes. It becomes clean. I noticed it, too.
After a two-hour cab ride, I arrived at the American consulate in Jerusalem, completed my interview and received my visa. I then had several days to sightsee.
I toured Jerusalem’s Old City and prayed in the Al-Aqsa Mosque — a site of frequent conflict. I visited the Western Wall and watched as people prayed.
Jerusalem was bubbling with life: music, spices, perfumes. The people were of many colors and faiths, and there were many pilgrims. In the Old City, all three faiths — Christianity, Judaism, Islam — commingled, with sacred sites sometimes just a few feet away from one another.
The next day I visited Haifa. It appeared less tense than Jerusalem, and in some places the mixing of different people was beautifully manifest. There you could use shalom and as-salamu alaykum interchangeably.
I also visited the Baha’i gardens, at the heart of which stands the golden-domed Shrine of the Báb, the resting place of the prophet-herald of the Baha’i faith. The gardens were immensely beautiful, but I was reminded that Gaza has only one public garden: the Unknown Soldier’s Square, where jobless and poor people sit and watch each other.
Then came my most poignant experience: A friend drove me to Tiberias, a small city on the shores of the Sea of Galilee. Starving after many hours without food, I ate dinner at a nearby kibbutz’s restaurant for tourists. When I asked why the streets were empty, I was reminded that it was the Jewish Sabbath.
I entered the restaurant and found people eating and laughing together. I was nervous; earlier in the day there had been tensions at the border, between Hamas and Israel.
But my curiosity kept me from leaving. I ate salmon and drank Pepsi. I worried about what everyone would do if they discovered I was a Palestinian. But nobody seemed suspicious.
I also thought about what Palestinians would say if they knew I entered and ate in such a place like a normal tourist. Some Palestinians, I thought, might see this as a crime of treason, a normalization with the “enemy.”
I exchanged smiles with several people. Maybe they thought I was Korean, as sometimes happens in Gaza.
What I noticed during my meal was a sense of harmony, love and unity. It was overwhelming. I saw maybe one or two tables of big families: children doing their homework, women chatting with each other.
Around this time, my friends and colleagues called to say that three babies had burned to death in their home at a refugee camp in Gaza. The fire was started by a candle that was being used during a power outage. I cried the next morning; it felt too normal.
I visited Israel’s border with Jordan and Syria. I saw ruins — of houses and of a mosque — that reminded me of Gaza. The buildings were left destroyed after the war between Israel and Syria. There was graffiti on the walls in Hebrew, Arabic and English: a big pink heart with “BFF” written inside, dedications and love letters, a sentence in Arabic declaring that ISIS had arrived here, Allahu akbar, and the word “Gaza.”
I interviewed one family who was visiting the place. It was empty and calm. The walls were black, as if they’d been set on fire. I wondered what had happened to the people who used to live and pray here.
Two days after I returned to Gaza, I visited the mobile home of a displaced family whose previous home had been destroyed during the last war.
It was close to the border fence.
The moment I entered, I thought of the Israeli kibbutz. There was a similar sense of harmony.
But here, in the “Caravan Quarter” of Beit Hanoun, in northern Gaza, there were no restaurants, no schools, no nature."""
texte3="bonjour."
t1=time.time()
test3=bert_poids_class.keywords(texte3)
test3.extraction()
test3.remplissage()

test3_bis=bert_class.article(texte3)
test3_bis.sent_embeddings()
test3_bis.article_embedding(test3.poids)
print("Temps 3 : {}".format(time.time()-t1))

def sim(a,b):
    return 1 - cosine(a,b)

print("sim 1 ", sim(test_bis.article_emb,test2_bis.article_emb))
#print("sim 2 ",sim(test_bis.article_emb,test3_bis.article_emb))
#print("sim 3 ", sim(test2_bis.article_emb,test3_bis.article_emb))


