import bert_class
import time
from scipy.spatial.distance import cosine

article = """OMAHA — The United States Olympic swimming trials are the spectacle that Michael Phelps built. If he had not strapped the sport to his broad back and climbed to the Olympic summit six times in Athens in 2004 and eight times in eight nights in Beijing in 2008, there would be no pre-finals light show at CenturyLink Center, no ticket scalpers. The crowds would be far smaller than 14,000, and fewer athletes would be extending their careers past college with the help of corporate endorsements.
And there would have been at least one fewer entrant in the 200-meter butterfly.
Phelps’s influence on the sport he set out to make more prominent, more professional, goes well beyond the eyeballs and the excitement he has brought to swimming since he appeared in his first United States trials, a much more subdued affair in Indianapolis in 2000. Phelps has inspired a generation to swim, including a few world-caliber athletes who crafted competitive schedules in Phelps’s image.
Nobody was Phelps’s equal in the 200-meter butterfly final on Wednesday. He qualified for his fifth Olympic team by winning in 1 minute 54.84 seconds, ahead of Tom Shields (1:55.81). Those in his wake included Chase Kalisz, 22, who finished fifth, having already earned a berth to the Rio Games with a victory in the 400-meter individual medley on Sunday.
Kalisz’s best leg in the I.M. is the breaststroke, and he could probably be world-class in that stroke’s events if he put his mind (and his training) to it. But Kalisz grew up in the Baltimore suburbs, and his second home was Meadowbrook Aquatic & Fitness Center, which Phelps had made famous as the most illustrious in a long line of star swimmers produced by North Baltimore Aquatic Club.
Kalisz wanted to be like Mike, so he made himself into a world-class butterfly swimmer. “Me doing butterfly is a testament to me wanting to swim the same event as Michael,” Kalisz said. “Butterfly didn’t come natural to me.”
He added: “Butterfly was something I had to work at, and I just loved racing next to Michael, and I did that enough that ultimately I would say the 200 butterfly is probably my second-best event. So I think me just wanting to emulate Michael so much is why my 2 fly is the way it is.”
The Phelps effect knows no geographical boundaries. At the 2012 Olympics, the South African Chad Le Clos swam the same four individual events as Phelps (the 100 and 200 butterfly and the 200 and 400 I.M.). After handing Phelps his first international defeat in 10 years in the 200 butterfly, Le Clos explained that he had added more events to his program, including the one that gave him a gold medal, after watching Phelps win a record eight golds in Beijing.
“He was the reason I swam the butterfly,” Le Clos said. “It’s not a joke.”
He added: “That’s why I swim the 200 freestyle, both the I.M.s. I don’t swim it for any other reason than just because Michael does.”
Le Clos took it personally when Phelps said last year that the men’s butterfly times around the world had been slow after the London Games, a factor that nudged Phelps out of retirement. After winning the 100 butterfly in 50.56 seconds at last summer’s world championships in Russia — a competition from which Phelps was absent — Le Clos crowed that his idol could “keep quiet now.”
Phelps, who competed at the United States senior nationals last summer after being removed from the American squad for the world championships following a drunken-driving arrest, responded hours later with a 50.45 to win the 100 butterfly.
“Chad liked me, and then he didn’t like me,” Phelps said recently with a laugh. “He said I was his hero, and then he was calling me out.”
In the winter of his career, Phelps finds himself in a position similar to that of Tiger Woods: The children he inspired have grown up to provide some of his fiercest competition.
At the 2014 Pan Pacific Championships in Australia, Daiya Seto of Japan shyly approached Phelps and showed him a photograph, taken at the 2001 edition of the meet in Japan. It was of a 7-year-old Seto, now one of the top individual medley swimmers in the world, posing with Phelps. “It was unbelievable,” Phelps said. “It’s insane.”
The Australian Mitch Larkin had not yet become a worldbeater in the backstroke events when he had a memorable encounter with Phelps. It was in 2012, at an Olympic tuneup meet in the United States. Larkin said he had been walking on the pool deck after racing in the 200 I.M. when Phelps called him by name, praised his effort and said, “Keep doing what you’re doing.”
Larkin said, “That was a massive moment for me.”
He said he had called his parents back in Australia, the time difference be damned, to tell them about the encouragement. “It just brings tears to my eyes remembering how excited Mitch was,” said Larkin’s mother, Judy. “It was a lovely moment for him that he’ll always treasure.”
Four years later, Larkin, 22, is the reigning world champion and the world No. 1 in the 100 and 200 backstrokes. Phelps’s magnanimous gesture, however indirectly, could end up spelling the end to the United States’ dominance in the stroke. The American men have won the last five Olympic gold medals awarded in the backstroke events.
The loss to Le Clos notwithstanding, Phelps has owned the 200 butterfly since placing fifth in the event at the 2000 Olympics at age 15. He has held the world record continuously since 2001, lowering it by 3.41 seconds in that span, to 1:51.51.
Between 2007 and 2009, his peak performance years, Phelps posted the four fastest times in history in the event. His 1:52.94 at last summer’s senior nationals led the world rankings in 2015.
The world has closed the gap, but Phelps remains the gold standard."""


t1=time.time()
test=bert_class.article(article)
test.sent_embeddings()   # c'est cette etape qui prend du temps! Logique.
test.article_embedding()
print(time.time()-t1)

"""t1=time.time()
test2=bert_class.article(article2)
test2.sent_embeddings()   # c'est cette etape qui prend du temps! Logique.
test2.article_embedding(-1)
print(time.time()-t1)

t1=time.time()
test3=bert_class.article(article3)
test3.sent_embeddings()   # c'est cette etape qui prend du temps! Logique.
test3.article_embedding(-1)
print(time.time()-t1)

def sim(a,b):
    return 1 - cosine(a,b)

print(sim(test.article_emb,test2.article_emb))
print(sim(test.article_emb,test3.article_emb))
print(sim(test2.article_emb,test3.article_emb))"""